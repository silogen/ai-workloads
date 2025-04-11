from evaluation_metrics.metrics.run_metrics_evaluation import run


def test_run_evaluation_single_correct_answer():
    data = [
        {
            "inference_result": "This is generated text.",
            "gold_standard_result": ["This is generated text."],
            "prompt": "Generate text.",
        }
    ]
    results = run(generations=data)
    assert len(results.prompts) == 1
    assert "Generate text." in results.prompts
    assert len(results.scores.f1_list) == 1
    assert results.scores.accuracy == 1.0
    assert results.scores.bleu_score == 1.0
    assert results.scores.precision_bert == 1.0
    assert results.scores.recall_bert == 1.0
    assert results.scores.f1_bert == 1.0


def test_run_evaluation_incorrect_answer():
    data = [
        {
            "inference_result": "This is incorrect text.",
            "gold_standard_result": ["This is generated text."],
            "prompt": "Generate text.",
        }
    ]
    results = run(generations=data)
    assert len(results.prompts) == 1
    assert "Generate text." in results.prompts
    assert len(results.scores.f1_list) == 1
    assert results.scores.accuracy == 0.0
    assert results.scores.bleu_score < 1.0
    assert results.scores.precision_bert < 1.0
    assert results.scores.recall_bert < 1.0
    assert results.scores.f1_bert < 1.0


def test_run_evaluation_multiple_data_points():
    data = [
        {
            "inference_result": "This is generated text.",
            "gold_standard_result": ["This is generated text."],
            "prompt": "Generate text.",
        },
        {
            "inference_result": "Another generated text.",
            "gold_standard_result": ["Another generated text."],
            "prompt": "Generate another text.",
        },
    ]
    results = run(generations=data)
    assert len(results.prompts) == 2
    assert "Generate text." in results.prompts
    assert "Generate another text." in results.prompts
    assert len(results.scores.f1_list) == 2
    assert results.scores.accuracy == 1.0
    assert results.scores.bleu_score == 1.0
    assert results.scores.precision_bert == 1.0
    assert results.scores.recall_bert == 1.0
    assert results.scores.f1_bert == 1.0


def test_run_evaluation_multiple_correct_answers():
    data = [
        {
            "inference_result": "This is generated text.",
            "gold_standard_result": ["This is generated text.", "This is another correct answer."],
            "prompt": "Generate text.",
        }
    ]
    try:
        run(generations=data)
    except NotImplementedError as e:
        assert str(e) == "Multiple correct answers"
    else:
        assert False, "Expected NotImplementedError"
