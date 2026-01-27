scoring_prompt = """
You are scoring ONE image using the rubric below. Output ONLY valid JSON matching the schema. No extra keys, no notes.

Step 0 — Core Interaction Set (max 3, MUST be UNIQUE + PRIMARY)
A “core interaction” is a visually obvious, intentional action between TWO DISTINCT visible entities, expressible as:
  Agent VERB Patient

PRIMARY-ACTION RULE (IMPORTANT — prevents secondary-action inflation)
- Prefer the highest-level, goal-defining event that best explains what’s happening (the “main verb”).
- Do NOT count a secondary/mechanism action if it is subordinate to a stronger action you already count in the same scene (e.g., holding/supporting/gripping/aiming/reaching/looking/standing-by) and it involves the same agent–patient pair.
- Examples (subordinate → exclude):
  - eating/biting/drinking → exclude “hold food/cup”
  - writing/signing/drawing → exclude “hold pen/paper”
  - tool-use (cutting/stirring/hitting/painting) → exclude “hold tool” and exclude “tool touches target” as a separate interaction
  - throwing/handing → exclude “hold object” if the transfer/throw is clearly depicted

- Only keep a secondary action if it is independently the main event (no clearer event is visible) or it involves a different patient/partner and would still be salient if the main event were removed.

Countable core verbs (examples, not exhaustive):
- dynamic: hit/kick/throw/push/pull/lift/carry/hand/give/feed/pour/cut/stir/open/close
- contact/social: hug/shake hands/touch/pet
- consumption: eat/bite/drink
- animal: investigate/bite/paw (if clearly directed)

Do NOT count (unless necessary to express the primary action):
- support/container/spatial: in/on/under/next to/behind/in front of
- attributes/part-whole: has/wearing
- static scene affordances: “cup on table”, “shirt on child”

Uniqueness rule (IMPORTANT):
- The core_interactions list must contain UNIQUE interactions.
- NEVER restate the same interaction with different wording.
- If only one unique primary interaction exists, CIC must be 1 and list length must be 1.

Measure 1 — CIC (Core Interaction Count) [0–3]
- CIC = number of UNIQUE PRIMARY core interactions present in the scene.
- If there are 3+ unique primary interactions, set CIC = 3 and list the 3 most visually obvious.
- If CIC = 0/1/2, the list length MUST equal CIC.
- Conservative counting: if an interaction is uncertain or mostly implied, do not count it.

Measure 2 — SEP (Separability) [0–2]
Score separability ONLY for the entities in your listed core_interactions.
- 0 = at least one required entity-pair is not reliably separable as AOIs (heavy occlusion/merging/tiny/one blob)
- 1 = separable but crowded/borderline
- 2 = clearly separable (distinct regions; fixations assignable with low ambiguity)

Measure 3 — DYN (Scene Dynamics) [0–2]
How “dynamic” is what’s happening between entities (vs mostly static/passive contact)?
- 0 = mostly static/passive (posing, sitting, holding with no clear action, looking, resting)
- 1 = some action but mild/slow/ambiguous (pointing, gentle touching/petting, reading/using object without clear change)
- 2 = clearly dynamic interaction(s) (force/motion/transfer/change is obvious: running/jumping, hitting/throwing, pushing/pulling, pouring/cutting, handing/giving, biting/eating, feeding, etc.)

Measure 4 — QLT (Image Quality) [0–1]
How reliable is this image for consistent action labeling and AOI marking (sharpness, resolution, visibility)?
- 0 = low quality / high noise (blurry, low-res, heavy compression, poor lighting, key entities small/occluded, causing likely mislabeling)
- 1 = clear / low noise (sharp enough that two raters would agree quickly; key entities and boundaries are easy to see)

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
  "DYN": <0-2>
  "QLT": <0-1>
}

Final self-check before output (do internally):
- core_interactions are UNIQUE and PRIMARY (no “hold” if it’s just a mechanism for a stronger action)
- len(core_interactions) == min(CIC, 3)
- story is 1–2 sentences and purely visual
"""
