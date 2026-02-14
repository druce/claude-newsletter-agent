"""Tests for tools/models.py — Pydantic output models for MCP tools."""
import pytest


class TestArticleSummaryModels:
    def test_article_summary_list_roundtrip(self):
        from tools.models import ArticleSummaryList

        data = {
            "results_list": [
                {"id": 1, "summary": "- Point one\n- Point two\n- Point three"},
                {"id": 2, "summary": "- Single point"},
            ]
        }
        obj = ArticleSummaryList.model_validate(data)
        assert len(obj.results_list) == 2
        assert obj.results_list[0].id == 1
        assert "Point one" in obj.results_list[0].summary

    def test_distilled_story_list_roundtrip(self):
        from tools.models import DistilledStoryList

        data = {
            "results_list": [
                {"id": 1, "short_summary": "OpenAI launches GPT-5 with reasoning."},
            ]
        }
        obj = DistilledStoryList.model_validate(data)
        assert obj.results_list[0].short_summary.startswith("OpenAI")


class TestCategoryModels:
    def test_topic_category_list(self):
        from tools.models import TopicCategoryList

        data = {"categories": ["AI Policy", "LLM Updates", "Funding Rounds"]}
        obj = TopicCategoryList.model_validate(data)
        assert len(obj.categories) == 3

    def test_topic_assignment(self):
        from tools.models import TopicAssignment

        obj = TopicAssignment.model_validate({"topic_title": "AI Policy"})
        assert obj.topic_title == "AI Policy"

    def test_dedupe_result_list(self):
        from tools.models import DedupeResultList

        data = {
            "results_list": [
                {"id": 1, "dupe_id": -1},
                {"id": 2, "dupe_id": 1},
            ]
        }
        obj = DedupeResultList.model_validate(data)
        assert obj.results_list[0].dupe_id == -1
        assert obj.results_list[1].dupe_id == 1


class TestSectionModels:
    def test_section_with_headlines(self):
        from tools.models import Section

        data = {
            "section_title": "AI Takes the Stage",
            "headlines": [
                {
                    "headline": "OpenAI launches GPT-5",
                    "rating": 9.2,
                    "prune": False,
                    "links": [{"site_name": "Reuters", "url": "https://reuters.com/1"}],
                },
            ],
        }
        obj = Section.model_validate(data)
        assert obj.section_title == "AI Takes the Stage"
        assert len(obj.headlines) == 1
        assert obj.headlines[0].links[0].site_name == "Reuters"

    def test_section_critique(self):
        from tools.models import SectionCritique

        data = {
            "coherence_score": 8.5,
            "quality_score": 7.2,
            "actions": [
                {"id": 1, "action": "keep"},
                {"id": 2, "action": "drop"},
                {"id": 3, "action": "rewrite", "rewritten_headline": "Better headline"},
                {"id": 4, "action": "move", "target_category": "Other News"},
            ],
        }
        obj = SectionCritique.model_validate(data)
        assert obj.coherence_score == 8.5
        assert len(obj.actions) == 4
        assert obj.actions[2].rewritten_headline == "Better headline"


class TestNewsletterModels:
    def test_newsletter_critique(self):
        from tools.models import NewsletterCritique

        data = {
            "overall_score": 8.1,
            "title_quality": 8.5,
            "structure_quality": 7.9,
            "section_quality": 8.2,
            "headline_quality": 7.8,
            "should_iterate": False,
            "critique_text": "Good newsletter overall.",
        }
        obj = NewsletterCritique.model_validate(data)
        assert obj.overall_score == 8.1
        assert obj.should_iterate is False

    def test_string_result(self):
        from tools.models import StringResult

        obj = StringResult.model_validate({"result": "AI Shakes Up Markets"})
        assert obj.result == "AI Shakes Up Markets"


class TestRatingFilterModels:
    def test_filter_result_list(self):
        from tools.models import FilterResultList

        data = {
            "results_list": [
                {"id": 1, "output": True},
                {"id": 2, "output": False},
            ]
        }
        obj = FilterResultList.model_validate(data)
        assert obj.results_list[0].output is True
        assert obj.results_list[1].output is False
