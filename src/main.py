"""
Copyright © Sr@1 2017, All rights reserved

Contains the complete end to end training and prediction
"""
from tasks.localizer import run_localizer
# from tasks.classifier import run_classifier

"""
    * Running localizer task
        * Archetecture of the localizer can be found at /model_archs/localizer_as_regr.py
        * Settings for the localizer task can be found at /tasks/localizer.py
"""
run_localizer()

# run_classifier()
