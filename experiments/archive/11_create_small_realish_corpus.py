"""Create a small realish local passage corpus for Iteration 5.

The corpus is generated locally from hand-written paragraph fragments. It is
less templated than the Iteration 4 topic corpus, but still deterministic and
small enough for local ColBERT/PDX experiments.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from utils_colbert import save_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "realish_corpus"
DOCUMENTS_PATH = OUTPUT_DIR / "documents.jsonl"
QUERIES_PATH = OUTPUT_DIR / "queries.jsonl"
QRELS_PATH = OUTPUT_DIR / "qrels.json"
TARGET_DOCUMENTS = 600
RELEVANT_PER_QUERY = 4


QUERY_SPECS = [
    {
        "topic": "colbert_maxsim",
        "query": "How does ColBERT MaxSim combine query token matches into a document score?",
        "sentences": [
            "A late-interaction retriever keeps a separate vector for each query token and each passage token.",
            "For every query token, ColBERT looks for the strongest matching token inside a candidate passage.",
            "The document score is produced by summing those per-token maxima rather than averaging a single vector.",
            "This scoring rule preserves local evidence such as a technical term matching a nearby passage phrase.",
            "Exact MaxSim becomes expensive when every query token must be compared with every passage token.",
            "Candidate generation is usually used to avoid running full token comparisons against the entire corpus.",
        ],
    },
    {
        "topic": "token_mapping",
        "query": "Why does a flat ColBERT token index need token to document metadata?",
        "sentences": [
            "A flat token index returns identifiers for individual embedding rows, not document identifiers.",
            "Each token-vector row therefore needs metadata recording the source document and token position.",
            "Without this mapping, retrieved token neighbors cannot be aggregated into passage-level scores.",
            "The metadata also helps debug whether a high-scoring document is supported by one token or many tokens.",
            "Packed embedding files store offsets so variable-length token matrices can be reconstructed.",
            "Document-level reranking depends on collecting all token hits that belong to the same passage.",
        ],
    },
    {
        "topic": "pdx_layout",
        "query": "What is the purpose of the PDX dimension oriented vector layout?",
        "sentences": [
            "PDX reorganizes vectors inside blocks so that values from the same dimension are stored together.",
            "Dimension-oriented storage can improve memory locality when search scans dimensions incrementally.",
            "The layout is intended to make pruning and distance accumulation friendlier to the cache hierarchy.",
            "A PDX block can be consumed by compiled search code without leaving vectors in ordinary row-major order.",
            "For token-vector experiments, every ColBERT token embedding can be materialized as a PDX row.",
            "The layout itself does not compute a ColBERT document score; aggregation is still a separate step.",
        ],
    },
    {
        "topic": "bond_pruning",
        "query": "How does BOND prune candidates during nearest neighbor search?",
        "sentences": [
            "BOND processes dimensions progressively and maintains bounds on candidate distances.",
            "A candidate can be removed when its partial distance cannot beat the current top-k threshold.",
            "The method is exact when the pruning rule never discards a vector that could still enter the result set.",
            "Vertically decomposed storage makes it natural to scan one dimension across many vectors.",
            "Branch-and-bound behavior is most useful when strong candidates create a tight threshold early.",
            "PDX-BOND combines the PDX layout with this dimension-by-dimension exact search strategy.",
        ],
    },
    {
        "topic": "adsampling",
        "query": "What does adaptive dimension sampling try to reduce in vector search?",
        "sentences": [
            "Adaptive dimension sampling estimates whether a vector comparison needs every coordinate.",
            "The algorithm samples dimensions and stops early when the remaining uncertainty is unlikely to matter.",
            "Its purpose is to reduce distance-comparison work while preserving enough ranking quality.",
            "Sampling-based methods should always be checked against an exact nearest-neighbor reference.",
            "The approach is related to pruning because both avoid unnecessary arithmetic on weak candidates.",
            "In high-dimensional embeddings, even modest savings per comparison can affect total search cost.",
        ],
    },
    {
        "topic": "candidate_reranking",
        "query": "Why combine PDX token retrieval with exact ColBERT reranking?",
        "sentences": [
            "PDX token retrieval can quickly gather documents that contain strong token-level matches.",
            "The approximate token aggregation is useful for candidate generation but does not fully reproduce MaxSim.",
            "Exact reranking recomputes the ColBERT score over selected candidate documents only.",
            "This two-stage design can keep the expressive late-interaction score while reducing full-corpus scoring.",
            "A candidate set must contain the exact top documents before reranking can recover the reference ranking.",
            "The experiment therefore measures candidate recall separately from final qrels metrics.",
        ],
    },
    {
        "topic": "retrieval_metrics",
        "query": "Which retrieval metrics are useful for evaluating candidate generation?",
        "sentences": [
            "Recall at k measures how many known relevant passages appear in the first k results.",
            "MRR at ten rewards rankings that place a relevant passage near the top of the list.",
            "Candidate-generation experiments also need recall against the exact MaxSim top-k ranking.",
            "A reranker can only recover documents that were included in the candidate pool.",
            "Manual qrels are useful for small smoke tests even when they are incomplete.",
            "Ranking agreement and relevance metrics answer different questions and should both be reported.",
        ],
    },
    {
        "topic": "embedding_export",
        "query": "How are PyLate ColBERT embeddings stored for reproducible experiments?",
        "sentences": [
            "The export step writes all token vectors into a single values matrix with float32 entries.",
            "Offsets mark where each document or query token matrix starts and ends.",
            "The metadata records document identifiers, query identifiers, shapes, and the model name.",
            "Saving packed embeddings avoids repeated neural encoding when only search code is changing.",
            "The same export format can be consumed by exact NumPy baselines and by PDX token indexes.",
            "Variable-length matrices are expected because each passage has a different number of tokens.",
        ],
    },
    {
        "topic": "data_warehouse",
        "query": "What happens in a data warehouse ETL pipeline before analytics queries run?",
        "sentences": [
            "An ETL pipeline extracts records from source systems and checks that required fields are present.",
            "Transformations standardize timestamps, clean inconsistent codes, and attach business keys.",
            "Loaded warehouse tables are often partitioned so analysts can scan recent data efficiently.",
            "Quality checks compare row counts and aggregates before a daily report is published.",
            "Late-arriving events may be reconciled in a separate correction job.",
            "The goal is to make analytical queries reliable without exposing every raw operational detail.",
        ],
    },
    {
        "topic": "cybersecurity_logs",
        "query": "How can security analysts use logs during incident investigation?",
        "sentences": [
            "An incident analyst starts by preserving authentication, endpoint, and network logs for the relevant window.",
            "Correlating timestamps can reveal whether a suspicious login was followed by file access or privilege changes.",
            "Log enrichment adds host names, user roles, and geolocation hints to make triage easier.",
            "Analysts often build a timeline before deciding whether containment is required.",
            "False positives are documented so the detection rule can be tuned later.",
            "A concise incident note records what happened, what evidence was checked, and what action was taken.",
        ],
    },
    {
        "topic": "library_archives",
        "query": "How do archivists organize letters and records for historical research?",
        "sentences": [
            "Archivists arrange collections so letters, ledgers, maps, and photographs keep their original context.",
            "A finding aid describes the creator, date range, physical boxes, and major subjects in a collection.",
            "Researchers use call numbers to request specific folders without handling unrelated material.",
            "Fragile documents may be digitized so readers can inspect them while the originals remain protected.",
            "Provenance matters because a document's meaning often depends on who created and preserved it.",
            "Careful cataloging turns scattered papers into evidence that historians can cite responsibly.",
        ],
    },
    {
        "topic": "rail_scheduling",
        "query": "What makes railway station scheduling difficult during disruptions?",
        "sentences": [
            "A station controller must assign platforms while delayed trains approach from different directions.",
            "One late arrival can block a platform that another service was scheduled to use minutes later.",
            "Dispatchers balance passenger connections, crew availability, and the need to keep tracks clear.",
            "Real-time announcements depend on schedule updates reaching information systems quickly.",
            "Recovery plans may skip stops or turn trains early to restore the published timetable.",
            "The schedule is a living plan rather than a static table when disruptions accumulate.",
        ],
    },
    {
        "topic": "weather_forecast",
        "query": "How do forecasters explain a cold front and changing local weather?",
        "sentences": [
            "A cold front marks the leading edge of cooler air pushing beneath warmer air near the surface.",
            "As the front passes, wind direction can shift and showers may develop along the boundary.",
            "Forecast discussions often mention pressure changes, cloud cover, and the timing of precipitation.",
            "Temperatures usually fall after the front, especially if clearing skies allow heat to escape overnight.",
            "Local terrain can make rainfall heavier in some neighborhoods than in nearby areas.",
            "The public forecast translates these atmospheric details into practical advice about the day ahead.",
        ],
    },
    {
        "topic": "cooking_bread",
        "query": "What steps matter when baking a simple loaf of bread?",
        "sentences": [
            "A bread dough begins with flour, water, yeast, and salt mixed until no dry patches remain.",
            "Resting the dough gives flour time to hydrate and makes kneading easier.",
            "Fermentation develops flavor while bubbles of gas slowly expand the dough.",
            "Shaping creates surface tension so the loaf rises upward instead of spreading flat.",
            "Steam in the first minutes of baking helps the crust stay flexible before it browns.",
            "The loaf should cool before slicing because the crumb continues to set after leaving the oven.",
        ],
    },
    {
        "topic": "football_pressing",
        "query": "What is high pressing in a football match?",
        "sentences": [
            "High pressing asks forwards and midfielders to challenge the opponent near their own goal.",
            "The team tries to force hurried passes before the opponent can build a stable attack.",
            "A pressing trap may leave one obvious passing lane open and then close it aggressively.",
            "The tactic requires coordination because one late runner can create space behind the press.",
            "Coaches often use pressing after losing possession to win the ball back quickly.",
            "Fatigue can reduce pressing intensity late in a match and expose the defensive line.",
        ],
    },
    {
        "topic": "travel_museum",
        "query": "How should a traveler plan a quiet day of museums and walking?",
        "sentences": [
            "A good museum day starts with timed tickets for the busiest exhibition and a flexible plan afterward.",
            "Walking between nearby galleries can reveal courtyards, bookshops, and small cafes that maps overlook.",
            "Leaving space for lunch prevents the itinerary from becoming a checklist of rooms.",
            "Travelers often enjoy a neighborhood more when they combine one major sight with several minor stops.",
            "Public transit is useful for the first and last legs while the middle of the day can be explored on foot.",
            "A quiet evening walk helps connect the separate museum visits into a memory of the city.",
        ],
    },
    {
        "topic": "urban_planning",
        "query": "What do planners consider when redesigning a neighborhood street?",
        "sentences": [
            "Street redesign begins with observations of crossings, curb space, bus stops, and delivery activity.",
            "Planners must balance vehicle movement with safer walking routes and places for people to wait.",
            "A narrower lane can calm traffic if emergency access and freight needs are still respected.",
            "Trees, lighting, and benches change how comfortable the street feels at different times of day.",
            "Public meetings reveal conflicts between residents, shop owners, cyclists, and transit riders.",
            "A pilot project lets the city test markings and barriers before rebuilding the street permanently.",
        ],
    },
    {
        "topic": "renewable_microgrid",
        "query": "How does a renewable microgrid balance solar power and battery storage?",
        "sentences": [
            "A microgrid controller watches solar production, battery charge, local demand, and the main grid connection.",
            "During sunny hours, extra generation can charge batteries instead of being exported immediately.",
            "When clouds reduce output, stored energy helps smooth the difference between supply and demand.",
            "Critical loads may be prioritized if the microgrid disconnects during an outage.",
            "Forecasts of weather and evening demand help decide whether to save battery capacity.",
            "The controller's goal is resilience and cost control rather than maximum generation at every moment.",
        ],
    },
    {
        "topic": "hospital_triage",
        "query": "What information is used in an emergency department triage note?",
        "sentences": [
            "A triage note records the patient's main concern, vital signs, arrival time, and immediate risks.",
            "The nurse documents symptoms in a structured way so the care team can prioritize attention.",
            "A clear note distinguishes chronic background problems from the reason for the current visit.",
            "Triage does not finish the diagnosis; it decides how urgently the patient needs evaluation.",
            "Communication is important when the waiting room is busy and conditions may change.",
            "The record supports continuity as the patient moves from intake to examination and treatment.",
        ],
    },
    {
        "topic": "classroom_feedback",
        "query": "How can teachers use formative feedback during a lesson?",
        "sentences": [
            "Formative feedback helps a teacher see what students understand before the final assessment.",
            "Short written responses, exit tickets, and class discussions can reveal misconceptions early.",
            "The teacher can adjust the next activity when many students struggle with the same idea.",
            "Feedback works best when it points to a specific next step rather than only a grade.",
            "Students also learn from comparing an example answer with the criteria for success.",
            "The purpose is to guide learning while there is still time to improve.",
        ],
    },
    {
        "topic": "architecture_daylight",
        "query": "Why do architects study daylight before designing interior spaces?",
        "sentences": [
            "Daylight studies show how sun angles and window placement affect a room throughout the year.",
            "Too much direct light can cause glare, while too little light makes a space feel closed and dim.",
            "Architects use overhangs, shelves, and reflective surfaces to distribute daylight more evenly.",
            "The analysis also affects energy use because lighting and cooling loads change with the facade.",
            "A comfortable workspace needs views and brightness without forcing people to close blinds all day.",
            "Daylight is both a technical constraint and a central part of the character of a building.",
        ],
    },
    {
        "topic": "music_rehearsal",
        "query": "What happens during a careful ensemble music rehearsal?",
        "sentences": [
            "An ensemble rehearsal often starts by tuning and agreeing on tempo for difficult passages.",
            "The director may isolate a short phrase so players can listen for balance and articulation.",
            "Musicians mark breathing, entrances, and dynamic changes directly into their parts.",
            "A good rehearsal alternates detailed correction with full runs so the piece keeps its shape.",
            "Listening across sections is as important as playing the right notes individually.",
            "The final performance depends on shared timing developed through repeated rehearsal.",
        ],
    },
    {
        "topic": "coffee_roasting",
        "query": "How does coffee roasting change flavor and aroma?",
        "sentences": [
            "Coffee roasting applies heat until green beans dry, yellow, brown, and release aromatic compounds.",
            "The roast profile controls how quickly temperature rises through each stage.",
            "A lighter roast can preserve acidity, while a darker roast emphasizes bitterness and body.",
            "Roasters listen for cracking sounds and watch color as clues about chemical change.",
            "Cooling the beans quickly stops the roast from drifting past the intended endpoint.",
            "Small adjustments in time and heat can make the same bean taste noticeably different.",
        ],
    },
    {
        "topic": "water_treatment",
        "query": "How is drinking water treated before it reaches homes?",
        "sentences": [
            "A water treatment plant removes sediment, filters fine particles, and disinfects the supply.",
            "Operators monitor turbidity, flow, chemical dosing, and storage levels throughout the day.",
            "Coagulation helps tiny particles clump together so they can settle or be filtered out.",
            "Disinfection protects the distribution network as water travels through pipes to customers.",
            "Routine sampling checks whether treated water meets quality standards.",
            "The process is designed to be reliable even when raw water conditions change after storms.",
        ],
    },
    {
        "topic": "household_budget",
        "query": "What does a practical household budget track each month?",
        "sentences": [
            "A household budget lists income, fixed bills, flexible spending, savings, and irregular expenses.",
            "Tracking categories over several months reveals where estimates differ from actual behavior.",
            "An emergency fund line can prevent small surprises from becoming high-interest debt.",
            "Budget reviews are easier when subscriptions and automatic payments are visible in one place.",
            "The goal is not perfect prediction but a plan that guides decisions before money is spent.",
            "Seasonal costs such as travel, repairs, or school fees should be planned before they arrive.",
        ],
    },
    {
        "topic": "theater_lighting",
        "query": "How does lighting shape a theater performance?",
        "sentences": [
            "Theater lighting guides attention, sets mood, and helps the audience understand shifts in place or time.",
            "Designers choose color, angle, brightness, and timing to support the director's interpretation.",
            "A cue sheet records exactly when each lighting change should occur during the performance.",
            "Side light can reveal movement while a narrow spotlight can isolate a single performer.",
            "Technical rehearsals coordinate lighting with sound, scenery, and stage management.",
            "Good lighting feels integrated with the story rather than like decoration placed on top of it.",
        ],
    },
    {
        "topic": "astronomy_observatory",
        "query": "Why do observatories care about sky conditions and instrument calibration?",
        "sentences": [
            "An observatory records sky brightness, humidity, seeing, and cloud cover before collecting data.",
            "Calibration frames help distinguish real celestial signals from detector noise and optical artifacts.",
            "Astronomers schedule sensitive observations when the target is high and the atmosphere is stable.",
            "A small pointing error can waste time if the instrument misses the intended field.",
            "Repeated measurements are compared so researchers can reject corrupted exposures.",
            "Careful observing logs make later analysis possible long after the night has ended.",
        ],
    },
    {
        "topic": "harbor_history",
        "query": "What can a historic harbor reveal about trade and city development?",
        "sentences": [
            "A historic harbor connects warehouses, customs records, ship repair yards, and nearby markets.",
            "The goods unloaded at the waterfront can show how a city was tied to distant regions.",
            "Changes in docks and rail links reveal shifts from sail to steam and from crates to containers.",
            "Harbor maps often preserve evidence of industries that disappeared from the modern shoreline.",
            "Immigration records and passenger lists add social history to the story of trade.",
            "Studying the harbor turns infrastructure into a record of economic and urban change.",
        ],
    },
    {
        "topic": "field_notes",
        "query": "What makes field notes useful after an observation session?",
        "sentences": [
            "Field notes capture time, place, conditions, observations, and the researcher's immediate interpretation.",
            "Good notes separate direct description from later guesses or explanations.",
            "Sketches, measurements, and short quotations can preserve details that memory would smooth over.",
            "A consistent format makes it easier to compare observations across multiple visits.",
            "The notes should record surprises as well as expected events because anomalies may become important.",
            "Careful documentation lets another reader understand how conclusions were reached.",
        ],
    },
    {
        "topic": "project_planning",
        "query": "How should a small research software project record progress?",
        "sentences": [
            "A research software project benefits from short notes that record commands, errors, and decisions.",
            "Small reproducible experiments make it easier to tell whether a change improved the system.",
            "The setup log should distinguish measured results from hypotheses about future work.",
            "Versioned scripts and stable artifact paths let teammates rerun the same experiment later.",
            "When an attempt fails, keeping the error message prevents the team from repeating the same path.",
            "Progress notes should end with the next concrete step rather than broad ambitions.",
        ],
    },
]


DISTRACTOR_BANKS = [
    [
        "The city council published a quiet notice about sidewalk repairs near the central square.",
        "Residents described the old market hall as useful but difficult to navigate during busy hours.",
        "A neighborhood newsletter listed opening times, volunteer shifts, and notes from the last meeting.",
        "The minutes were written plainly, with names removed and decisions grouped by topic.",
        "Several shop owners asked for clearer signs before the seasonal festival begins.",
        "A short inspection report mentioned loose paving stones and faded lane markings.",
    ],
    [
        "The garden journal recorded when seedlings were moved outside and which beds received compost.",
        "A long dry spell changed the watering schedule for the greenhouse and raised beds.",
        "The caretaker noted that shaded corners stayed damp even after the paths had dried.",
        "Several volunteers repaired the tool shed before labeling shelves for the summer season.",
        "The planting plan grouped herbs near the kitchen door and taller stems along the wall.",
        "A hand-drawn map showed paths, benches, and the small storage area beside the fence.",
    ],
    [
        "The workshop checklist began with protective eyewear, clean benches, and a count of missing tools.",
        "A technician replaced the worn belt and wrote the part number in the maintenance notebook.",
        "Dust collection improved after the filter was cleaned and the hose connection was tightened.",
        "The supervisor asked everyone to report unusual noise before a machine failed completely.",
        "Measurements were written twice because the first board had warped overnight.",
        "A careful setup saved time when the same cut had to be repeated for multiple panels.",
    ],
    [
        "The regional newsletter described a new bus route that would connect the campus and hospital.",
        "Several passengers said the old timetable made evening transfers difficult after work.",
        "Planners counted boardings at each stop before deciding where shelters should be installed.",
        "The first week of service included staff at major stops to answer route questions.",
        "A printed map helped riders understand how the loop changed during construction.",
        "The schedule was adjusted after drivers reported congestion near the bridge.",
    ],
    [
        "The festival program placed lectures in the morning and concerts after sunset.",
        "Volunteers checked tickets, directed guests, and kept the narrow hallway from becoming crowded.",
        "A small exhibition displayed posters, photographs, and handwritten notes from earlier events.",
        "The organizer kept a spare room available in case rain forced an outdoor session inside.",
        "Local cafes extended hours because visitors stayed in the district after the final performance.",
        "The closing remarks thanked technicians whose work was mostly invisible to the audience.",
    ],
    [
        "The office move required numbered crates, floor plans, and a list of equipment that needed special handling.",
        "Employees were asked to label cables before monitors and docking stations were packed.",
        "A temporary reception desk kept the service open while the front room was repainted.",
        "The move coordinator scheduled noisy work before client meetings began for the day.",
        "Several teams used the relocation to discard outdated folders and duplicate supplies.",
        "The new layout placed shared printers near the corridor instead of beside individual desks.",
    ],
    [
        "The local history display used maps to show how streets changed after the old canal was filled.",
        "Visitors paused over photographs that showed the same corner before and after the station opened.",
        "A caption explained why several houses were rebuilt with narrower fronts after a tax change.",
        "The curator added a listening station with recorded memories from former factory workers.",
        "School groups were given a worksheet that asked them to compare two different decades.",
        "The exhibition ended with a timeline of fires, floods, elections, and construction projects.",
    ],
    [
        "The hiking club posted route notes that included water sources, steep sections, and return options.",
        "Clouds gathered near the ridge, so the leader shortened the planned loop before lunch.",
        "A map and compass were kept ready even though the trail was marked clearly at the start.",
        "The group checked footwear and layers before leaving the parking area.",
        "Loose stones made the descent slower than the climb despite the shorter distance.",
        "The trip report listed travel time, rest stops, and conditions for the next group.",
    ],
    [
        "The restaurant manager reviewed reservations, deliveries, and staff assignments before service.",
        "A prep list on the wall divided sauces, vegetables, and desserts by station.",
        "The dining room was reset after a large party moved two tables together.",
        "A supplier called about a late delivery, so the menu board was changed before opening.",
        "The closing checklist included refrigeration logs and a count of linens for the laundry.",
        "Regular customers noticed the new lunch menu but still ordered their usual soup.",
    ],
    [
        "The community workshop introduced basic spreadsheet formulas with a budget example.",
        "Participants practiced sorting rows, freezing headers, and checking totals with simple functions.",
        "The instructor explained that a clean table is easier to analyze than a decorated one.",
        "Several attendees brought laptops with different keyboard layouts, which slowed the first exercise.",
        "A printed handout summarized shortcuts and common mistakes to review at home.",
        "The final task asked everyone to create a small chart from the same sample data.",
    ],
]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def relevant_passage(spec: dict, passage_index: int) -> str:
    sentences = spec["sentences"]
    start = (passage_index * 2) % len(sentences)
    chosen = [
        sentences[start],
        sentences[(start + 1) % len(sentences)],
        sentences[(start + 3) % len(sentences)],
    ]
    if passage_index % 2 == 0:
        chosen.append(
            "The passage emphasizes the practical details that a retrieval system should preserve."
        )
    else:
        chosen.append(
            "These details make the passage a useful target for a focused information need."
        )
    return " ".join(chosen)


def near_miss_passage(spec: dict, passage_index: int) -> str:
    sentences = spec["sentences"]
    chosen = [
        sentences[(passage_index + 1) % len(sentences)],
        sentences[(passage_index + 4) % len(sentences)],
    ]
    return (
        "This background note mentions related terminology but does not answer the query directly. "
        + " ".join(chosen)
    )


def distractor_passage(rng: random.Random, doc_index: int) -> tuple[str, str]:
    bank_index = doc_index % len(DISTRACTOR_BANKS)
    bank = DISTRACTOR_BANKS[bank_index]
    chosen = rng.sample(bank, 3)
    lead = [
        "The passage is taken from a local planning note.",
        "The short report describes an ordinary administrative detail.",
        "The paragraph records a practical observation from a community setting.",
        "The note reads like a small item from a local bulletin.",
    ][doc_index % 4]
    return f"distractor_{bank_index:02d}", f"{lead} {' '.join(chosen)}"


def make_documents() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, list[str]]]:
    rng = random.Random(20260525)
    documents = []
    queries = []
    qrels = {}

    for query_index, spec in enumerate(QUERY_SPECS, start=1):
        query_id = f"rq{query_index:03d}"
        topic_slug = slug(spec["topic"])
        queries.append(
            {
                "id": query_id,
                "text": spec["query"],
                "topic": spec["topic"],
            }
        )
        qrels[query_id] = []

        for passage_index in range(1, RELEVANT_PER_QUERY + 1):
            doc_id = f"rel-{topic_slug}-{passage_index:02d}"
            qrels[query_id].append(doc_id)
            documents.append(
                {
                    "id": doc_id,
                    "topic": spec["topic"],
                    "kind": "relevant",
                    "text": relevant_passage(spec, passage_index),
                }
            )

        for passage_index in range(1, 3):
            documents.append(
                {
                    "id": f"near-{topic_slug}-{passage_index:02d}",
                    "topic": spec["topic"],
                    "kind": "near_miss",
                    "text": near_miss_passage(spec, passage_index),
                }
            )

    distractor_index = 1
    while len(documents) < TARGET_DOCUMENTS:
        topic, text = distractor_passage(rng, distractor_index)
        documents.append(
            {
                "id": f"dist-{distractor_index:04d}",
                "topic": topic,
                "kind": "distractor",
                "text": text,
            }
        )
        distractor_index += 1

    rng.shuffle(documents)
    return documents, queries, qrels


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    documents, queries, qrels = make_documents()

    write_jsonl(DOCUMENTS_PATH, documents)
    write_jsonl(QUERIES_PATH, queries)
    save_json(QRELS_PATH, qrels)

    print("Created small realish corpus")
    print(f"documents: {len(documents)}")
    print(f"queries: {len(queries)}")
    print(f"qrels labels: {sum(len(ids) for ids in qrels.values())}")
    print(f"documents path: {DOCUMENTS_PATH}")
    print(f"queries path: {QUERIES_PATH}")
    print(f"qrels path: {QRELS_PATH}")


if __name__ == "__main__":
    main()
