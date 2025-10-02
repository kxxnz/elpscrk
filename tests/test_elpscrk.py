import os
import tempfile
import json
from elpscrk import WordlistGenerator, MarkovModel, TemplateExpander, BreachList, KeyboardMutator, nist_entropy_estimate


def test_extract_tokens_and_generate():
    profile = {'names': ['Ana Maria'], 'usernames': ['anam'], 'corpus_text': 'ana loves cats'}
    gen = WordlistGenerator(profile, max_candidates=200, min_entropy=10.0)
    gen.attach_template_expander(TemplateExpander())
    ranked = gen.generate()
    assert isinstance(ranked, list)
    assert len(ranked) > 0


def test_markov_train_and_generate():
    m = MarkovModel(order=2)
    m.train('password\npass\nword')
    m.set_sampling(enabled=False)
    out = m.generate(n=5)
    assert isinstance(out, list)


def test_breach_loader_and_freq(tmp_path):
    p = tmp_path / 'breach.txt'
    p.write_text('123456\n1000 password\npass 500')
    b = BreachList(str(p))
    assert b.get_freq('123456') is not None
    assert b.get_freq('password') == 1000


def test_keyboard_mutator():
    kb = KeyboardMutator()
    out = kb.mutate('test', depth=1, max_variants=5)
    assert isinstance(out, set)


def test_nist_entropy():
    e = nist_entropy_estimate('Ab1!')
    assert e > 0
