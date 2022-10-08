The code in this folder was used to produce the data analyzed for the journal article. This code has been tested on Python 3.8.5.

The few outside libraries brought in aren't used in version-specific manners, so nearly any version compatible with Python 3.8.5 should be fine.
However, if one wants to be exact, The versions of the libraries used in the experiment environment are as follows:
  -- numpy 1.19.1
  -- scipy 1.5.0
  -- wandb 0.12.3  (unused by the experiment, but code remains which references it.)
  -- pandas 1.1.3
  -- gym 0.17.3

All other code is included in this directory.

Usage:

To generate the data that was used in the article:
  1. Modify experiment_fullRun.py, line 1. Set the mainExperimentFolder variable to a valid path on your system where you want data to be stored, preferrably an empty folder.
  2. Run experiment_fullRun.py

The data will resemble, but not directly duplicate, that which was found in the article. Randomness plays a large role in CGP and RL.

Collecting the full set of data can take months and runs in a multi-threaded manner which can starve other processes of processor time. Use caution when
starting this experiment on a system for which you require continued responsiveness.

The data collected and analyzed by this article is included in this folder.