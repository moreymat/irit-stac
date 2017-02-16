#!/usr/bin/env python
# pylint: disable=invalid-name
# this is a script, not a module
# pylint: enable=invalid-name

"""
Learn and predict dialogue acts from EDU feature vectors
"""

import argparse
import copy
import os

import joblib
from scipy.sparse import lil_matrix
from sklearn.datasets import load_svmlight_file
from attelo.io import (load_labels,
                       load_vocab)

from educe.stac.annotation import set_addressees
from educe.stac.context import Context
from educe.stac.learning.addressee import guess_addressees_for_edu
import educe.stac.learning.doc_vectorizer as stac_vectorizer
import educe.stac
from educe.stac.util.output import save_document


# ---------------------------------------------------------------------
# learning
# ---------------------------------------------------------------------


def learn_and_save(learner, feature_path, output_path):
    """Learn dialogue acts from an svmlight features file and dump
    the model to disk.

    Parameters
    ----------
    learner : TODO
        Learner.

    feature_path : string
        Path to the sparse feature file in the svmlight format.

    output_path : string
        Path to which the model will be dumped.
    """
    # pylint: disable=unbalanced-tuple-unpacking
    data, target = load_svmlight_file(feature_path)
    # pylint: enable=unbalanced-tuple-unpacking
    output_dir = os.path.dirname(output_path)
    model = learner.fit(data, target)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    joblib.dump(model, output_path)


# ---------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------


def _output_key(key):
    """
    Given a `FileId` key for an input document, return a version that
    would be appropriate for its output equivalent
    """
    key2 = copy.copy(key)
    key2.stage = 'units'
    key2.annotator = 'simple-da'
    return key2


def get_edus_plus(inputs):
    """Generate EDUs and extra environmental information for each EDU.

    Parameters
    ----------
    inputs : FeatureInput
        Global information used for feature extraction.

    Yields
    ------
    env : DocEnv
        Document environment (for feature extraction), for the next EDU.

    contexts : educe.stac.context.Context
        Context for the next EDU, i.e. essentially a collection of
        pointers to the turn, tstar, turn_edus, dialogue,
        dialogue_turns, doc_turns, tokens.

    unit : educe.glozz.Unit
        Next EDU in the designated set of documents.
    """
    for env in stac_vectorizer.mk_envs(inputs, 'unannotated'):
        doc = env.current.doc
        contexts = Context.for_edus(doc)
        for unit in doc.units:
            if educe.stac.is_edu(unit):
                yield env, contexts, unit


def extract_features(vocab, edus_plus):
    """Return a sparse matrix of features for all EDUs in the corpus.

    Parameters
    ----------
    vocab : dict(string, int)
        Feature vocabulary.

    edus_plus : list((env, contexts, unit))
        List of EDUs with the relevant context for feature extraction.

    Returns
    -------
    matrix : scipy.sparse.csr_matrix
        Feature matrix in the Compressed Sparse Row format.
    """
    matrix = lil_matrix((len(edus_plus), len(vocab)))
    # this unfortunately duplicates
    # `educe.stac.learning.doc_vectorizer.extract_single_features()`
    # but it's the price we pay to ensure we get the edus and vectors in
    # the same order
    for row, (env, _, edu) in enumerate(edus_plus):
        vec = stac_vectorizer.SingleEduKeys(env.inputs)
        vec.fill(env.current, edu)
        for feat, val in vec.one_hot_values_gen():
            if feat in vocab:
                matrix[row, vocab[feat]] = val
    return matrix.tocsr()


def annotate_edus(model, vocab, labels, inputs):
    """Annotate each EDU with its dialogue act and addressee.

    This modifies the EDUs in memory ; use `save_document` afterwards
    to dump the modified annotations.

    Parameters
    ----------
    model : TODO
        TODO

    vocab : dict(string, int)
        Feature vocabulary.

    labels : list of string
        Array of labels.

    inputs : FeatureInput
        Global information for feature extraction.
    """
    edus_plus = list(get_edus_plus(inputs))
    feats = extract_features(vocab, edus_plus)
    predictions = model.predict(feats)
    for (env, contexts, edu), da_num in zip(edus_plus, predictions):
        da_label = labels[int(da_num) - 1]
        addressees = guess_addressees_for_edu(contexts,
                                              env.current.players,
                                              edu)
        set_addressees(edu, addressees)
        edu.type = da_label


def command_annotate(args):
    """
    Top-level command: given a dialogue act model, and a corpus with some
    Glozz documents, perform dialogue act annotation on them, and simple
    addressee detection, and dump Glozz documents in the output directory
    """
    args.ignore_cdus = False
    args.parsing = True
    args.single = True
    args.strip_mode = 'head'  # FIXME should not be specified here
    inputs = stac_vectorizer.read_corpus_inputs(args)
    model = joblib.load(args.model)
    vocab = {f: i for i, f in
             enumerate(load_vocab(args.vocabulary))}
    labels = load_labels(args.labels)
    labels = labels[1:]  # (0, __UNK__) was not explicitly stored before
    # add dialogue acts and addressees
    annotate_edus(model, vocab, labels, inputs)

    # corpus has been modified in-memory, now save to disk
    for key in inputs.corpus:
        key2 = _output_key(key)
        doc = inputs.corpus[key]
        save_document(args.output, key2, doc)


def main():
    "channel to subcommands"

    psr = argparse.ArgumentParser(add_help=False)
    psr.add_argument("corpus", default=None, metavar="DIR",
                     help="corpus to annotate (live mode assumed)")
    psr.add_argument('resources', metavar='DIR',
                     help='Resource dir (eg. data/resources/lexicon)')
    psr.add_argument("--model", default=None, required=True,
                     help="provide saved model for prediction of "
                     "dialogue acts")
    psr.add_argument("--vocabulary", default=None, required=True,
                     help="feature vocabulary")
    psr.add_argument("--labels", default=None, required=True,
                     help="dialogue act labels file")
    psr.add_argument("--output", "-o", metavar="DIR",
                     default=None,
                     required=True,
                     help="output directory")
    psr.set_defaults(func=command_annotate)

    args = psr.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

# vim:filetype=python:
