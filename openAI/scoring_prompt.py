scoring_prompt = """
You are scoring ONE image using the rubric below. Output ONLY valid JSON matching the schema. No extra keys, no notes.

Step 0 — Core Interaction Set (max 3, MUST be UNIQUE)
A “core interaction” is a visually obvious, intentional action between TWO DISTINCT visible entities, expressible as:
  Agent VERB Patient

Countable core verbs include: hold/carry/lift/push/pull, pour/cut/stir/throw, ride/drive/kick/hit, touch/hug/handshake, investigate/bite/paw (if clearly directed).

Do NOT count (unless it’s necessary to express the action):
- support/container/spatial: in/on/under/next to/behind/in front of
- attributes/part-whole: has/wearing
- static scene affordances: “cup on table”, “shirt on child”

Mechanism rule (IMPORTANT):
- If multiple descriptions refer to the SAME underlying action, DO NOT split them.
  Prefer the highest-level interaction.
  Example: tennis → use “person hits ball” (or “person hits ball with racket”), NOT “person holds racket” + “racket hits ball”.
  Example: signing → use “person signs bat”, NOT “person holds bat” (unless holding is the only clear action).

Uniqueness rule (IMPORTANT):
- The core_interactions list must contain UNIQUE interactions.
- NEVER repeat the same interaction with different wording to inflate counts.
- If only one unique interaction exists, CIC must be 1 and the list must have length 1.

Measure 1 — CIC (Core Interaction Count) [0–3]
- CIC = number of UNIQUE core interactions present in the scene.
- If there are 3+ unique interactions, set CIC = 3 and list the 3 most visually obvious.
- If CIC = 0/1/2, the list length MUST equal CIC.

Measure 2 — SEP (Separability) [0–2]
Score separability ONLY for the entities in your listed core_interactions.
- 0 = at least one required entity-pair is not reliably separable as AOIs (heavy occlusion/merging/tiny/one blob)
- 1 = separable but crowded/borderline
- 2 = clearly separable (distinct regions; fixations assignable with low ambiguity)

Measure 3 — CLR (Depictability / Agreement) [0–2]
How unambiguous are the listed core interactions from pixels alone (no story knowledge)?
- 0 = ambiguous/inferential (a naive rater may disagree on the interaction)
- 1 = mostly clear, but at least one interaction is borderline
- 2 = very clear, visually obvious interaction(s) (naive raters would agree quickly)

Measure 4 — PRM (Interaction Prominence) [0–2]
Are the interacting entities prominent enough for reliable analysis?
- 0 = core entities are tiny/distant/partially hidden or visually dominated by clutter
- 1 = usable but not ideal (moderate size or some crowding)
- 2 = prominent/central/large enough to see interaction clearly

Additional field — STORY (1–2 sentences)
Write a short, story-like description of what is happening, focusing on the primary visible actors and their actions.
- 1–2 sentences only
- Describe only what is visible (avoid speculation about intentions or unseen causes)
- Keep it plain and concrete

Output JSON ONLY with:
{
  "image_id": "<filename or id>",
  "core_interactions": [
    {"agent": "<short noun>", "verb": "<verb>", "patient": "<short noun>"}
  ],
  "story": "<1-2 sentences>",
  "CIC": <0-3>,
  "SEP": <0-2>,
  "CLR": <0-2>,
  "PRM": <0-2>
}

Final self-check before output (do internally):
- core_interactions are UNIQUE
- len(core_interactions) == min(CIC, 3)
- CIC is not inflated by decomposing one action into tool/mechanism sub-actions
- story is 1–2 sentences and purely visual

"""
