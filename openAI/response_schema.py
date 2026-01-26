response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "interaction_scoring_v2",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string"},
                "core_interactions": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "agent": {"type": "string"},
                            "verb": {"type": "string"},
                            "patient": {"type": "string"},
                        },
                        "required": ["agent", "verb", "patient"],
                        "additionalProperties": False,
                    },
                },
                "story": {"type": "string", "maxLength": 300},
                "CIC": {"type": "integer", "minimum": 0, "maximum": 3},
                "SEP": {"type": "integer", "minimum": 0, "maximum": 2},
                "CLR": {"type": "integer", "minimum": 0, "maximum": 2},
                "PRM": {"type": "integer", "minimum": 0, "maximum": 2},
            },
            "required": [
                "image_id",
                "core_interactions",
                "story",
                "CIC",
                "SEP",
                "CLR",
                "PRM",
            ],
            "additionalProperties": False,
        },
    },
}
