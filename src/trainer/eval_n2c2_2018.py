from __future__ import division

import argparse
import glob
import os
from collections import defaultdict
from xml.etree import cElementTree

class ClinicalCriteria(object):
    """Criteria in the Track 1 documents."""

    def __init__(self, tid, value):
        """Init."""
        self.tid = tid.strip().upper()
        self.ttype = self.tid
        self.value = value.lower().strip()

    def equals(self, other, mode='strict'):
        """Return whether the current criteria is equal to the one provided."""
        if other.tid == self.tid and other.value == self.value:
            return True
        return False


class RecordTrack1(object):
    """Record for Track 2 class."""

    def __init__(self, raw_annots):
        # raw_annotations: dict<string, string> -> Dictionary of Tag to Met/Unmet
        self.annotations = self._get_annotations(raw_annots)
        self.text = None

    @property
    def tags(self):
        return self.annotations['tags']

    def _get_annotations(self, raw_annots):
        """Return a dictionary with all the annotations in the .ann file."""
        annotations = defaultdict(dict)
        for tag, label in raw_annots.items():
            criterion = ClinicalCriteria(tag.upper(), label)
            annotations['tags'][tag.upper()] = criterion
        return annotations


class Measures(object):
    """Abstract methods and var to evaluate."""

    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        """Initizialize."""
        assert type(tp) == int
        assert type(tn) == int
        assert type(fp) == int
        assert type(fn) == int
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def precision(self):
        """Compute Precision score."""
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return 0.0

    def recall(self):
        """Compute Recall score."""
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return 0.0

    def f_score(self, beta=1):
        """Compute F1-measure score."""
        assert beta > 0.
        try:
            num = (1 + beta**2) * (self.precision() * self.recall())
            den = beta**2 * (self.precision() + self.recall())
            return num / den
        except ZeroDivisionError:
            return 0.0

    def f1(self):
        """Compute the F1-score (beta=1)."""
        return self.f_score(beta=1)

    def specificity(self):
        """Compute Specificity score."""
        try:
            return self.tn / (self.fp + self.tn)
        except ZeroDivisionError:
            return 0.0

    def sensitivity(self):
        """Compute Sensitivity score."""
        return self.recall()

    def auc(self):
        """Compute AUC score."""
        return (self.sensitivity() + self.specificity()) / 2


class MultipleEvaluator(object):
    """Evaluate two sets of files."""

    def __init__(self, corpora, tag_type=None, mode='strict',
                 verbose=False):
        """Initialize."""
        assert isinstance(corpora, Corpora)
        assert mode in ('strict', 'lenient')
        self.scores = None
        self.track1(corpora)

    def track1(self, corpora):
        """Compute measures for Track 1."""
        self.tags = ('ABDOMINAL', 'ADVANCED-CAD', 'ALCOHOL-ABUSE',
                     'ASP-FOR-MI', 'CREATININE', 'DIETSUPP-2MOS',
                     'DRUG-ABUSE', 'ENGLISH', 'HBA1C', 'KETO-1YR',
                     'MAJOR-DIABETES', 'MAKES-DECISIONS', 'MI-6MOS')
        self.scores = defaultdict(dict)
        metrics = ('p', 'r', 'f1', 'specificity', 'auc')
        values = ('met', 'not met')
        self.values = {'met': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
                       'not met': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}}

        def evaluation(corpora, value, scores):
            predictions = defaultdict(list)
            for g, s in corpora.docs:
                for tag in self.tags:
                    predictions[tag].append(
                        (g.tags[tag].value == value, s.tags[tag].value == value))
            for tag in self.tags:
                # accumulate for micro overall measure
                self.values[value]['tp'] += predictions[tag].count((True, True))
                self.values[value]['fp'] += predictions[tag].count((False, True))
                self.values[value]['tn'] += predictions[tag].count((False, False))
                self.values[value]['fn'] += predictions[tag].count((True, False))

                # compute per-tag measures
                measures = Measures(tp=predictions[tag].count((True, True)),
                                    fp=predictions[tag].count((False, True)),
                                    tn=predictions[tag].count((False, False)),
                                    fn=predictions[tag].count((True, False)))
                scores[(tag, value, 'p')] = measures.precision()
                scores[(tag, value, 'r')] = measures.recall()
                scores[(tag, value, 'f1')] = measures.f1()
                scores[(tag, value, 'specificity')] = measures.specificity()
                scores[(tag, value, 'auc')] = measures.auc()
            return scores

        self.scores = evaluation(corpora, 'met', self.scores)
        self.scores = evaluation(corpora, 'not met', self.scores)

        for measure in metrics:
            for value in values:
                self.scores[('macro', value, measure)] = sum(
                    [self.scores[(t, value, measure)] for t in self.tags]) / len(self.tags)


class Corpora(object):
    def __init__(self, golds, hyps):
        """Initialize."""
        self.golds, self.hyps = golds, hyps
        self.docs = []
        for g_annot, s_annot in zip(self.golds, self.hyps):
            g = RecordTrack1(g_annot)
            s = RecordTrack1(s_annot)
            self.docs.append((g, s))
            

def evaluate(corpora, mode='strict', verbose=False):
    assert mode in ('strict', 'lenient')
    evaluator_s = MultipleEvaluator(corpora, verbose)
    
    macro_f1, macro_auc = 0, 0
    print('{:*^96}'.format(' TRACK 1 '))
    print('{:20}  {:-^30}    {:-^22}    {:-^14}'.format('', ' met ',
                                                        ' not met ',
                                                        ' overall '))
    print('{:20}  {:6}  {:6}  {:6}  {:6}    {:6}  {:6}  {:6}    {:6}  {:6}'.format(
        '', 'Prec.', 'Rec.', 'Speci.', 'F(b=1)', 'Prec.', 'Rec.', 'F(b=1)', 'F(b=1)', 'AUC'))
    for tag in evaluator_s.tags:
        print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}'.format(
            tag.capitalize(),
            evaluator_s.scores[(tag, 'met', 'p')],
            evaluator_s.scores[(tag, 'met', 'r')],
            evaluator_s.scores[(tag, 'met', 'specificity')],
            evaluator_s.scores[(tag, 'met', 'f1')],
            evaluator_s.scores[(tag, 'not met', 'p')],
            evaluator_s.scores[(tag, 'not met', 'r')],
            evaluator_s.scores[(tag, 'not met', 'f1')],
            (evaluator_s.scores[(tag, 'met', 'f1')] + evaluator_s.scores[(tag, 'not met', 'f1')])/2,
            evaluator_s.scores[(tag, 'met', 'auc')]))
        macro_f1 += (evaluator_s.scores[(tag, 'met', 'f1')] + evaluator_s.scores[(tag, 'not met', 'f1')])/2
        macro_auc += evaluator_s.scores[(tag, 'met', 'auc')]
    print('{:20}  {:-^30}    {:-^22}    {:-^14}'.format('', '', '', ''))
    m = Measures(tp=evaluator_s.values['met']['tp'],
                 fp=evaluator_s.values['met']['fp'],
                 fn=evaluator_s.values['met']['fn'],
                 tn=evaluator_s.values['met']['tn'])
    nm = Measures(tp=evaluator_s.values['not met']['tp'],
                  fp=evaluator_s.values['not met']['fp'],
                  fn=evaluator_s.values['not met']['fn'],
                  tn=evaluator_s.values['not met']['tn'])
    print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}'.format(
        'Overall (micro)', m.precision(), m.recall(), m.specificity(),
        m.f1(), nm.precision(), nm.recall(), nm.f1(),
        (m.f1() + nm.f1()) / 2, m.auc()))
    print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}'.format(
        'Overall (macro)',
        evaluator_s.scores[('macro', 'met', 'p')],
        evaluator_s.scores[('macro', 'met', 'r')],
        evaluator_s.scores[('macro', 'met', 'specificity')],
        evaluator_s.scores[('macro', 'met', 'f1')],
        evaluator_s.scores[('macro', 'not met', 'p')],
        evaluator_s.scores[('macro', 'not met', 'r')],
        evaluator_s.scores[('macro', 'not met', 'f1')],
        macro_f1 / len(evaluator_s.tags),
        evaluator_s.scores[('macro', 'met', 'auc')]))
    print()
    print('{:>20}  {:^74}'.format('', '  {} files found  '.format(len(corpora.docs))))
    
    return {
        'micro-f1': (m.f1() + nm.f1()) / 2,
        'macro-f1': macro_f1 / len(evaluator_s.tags),
        'micro-auc': m.auc(),
        'macro-auc': evaluator_s.scores[('macro', 'met', 'auc')]
    }

            
def eval_n2c2_2018(golds, hyps, verbose=False):
    corpora = Corpora(golds, hyps)
    return evaluate(corpora, verbose=verbose)