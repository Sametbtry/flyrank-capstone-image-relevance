from pydantic import BaseModel, Field

class Metadata(BaseModel):
    category: str                                    # main entity in the image (e.g. "red_fox", "wolf", "moose")
    attributes: list[str]                           # descriptive traits or details
    caption: str                                    # readable image description
    confidence: float = Field(ge=0.0, le=1.0)       # model certainty score from 0-1


class BlogPostSeed(BaseModel):
    id: str                                         # stable post identifier (e.g. "post_01")
    title: str                                      # blog post headline
    content: str                                    # full post body used for embedding & subject extraction
    expected_category: str | None = None            # eval-only ground truth ("red_fox", "wolf", etc., or None)
    test_purpose: str                               # eval-only scenario label (e.g. probe_2_and_3, probe_4)