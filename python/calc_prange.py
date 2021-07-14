import pandas as pd
import numpy as np

#Compute pseudoranges and times
def compute_times(gnssRaw, gnssAnalysis):
  """ Shubh wrote this similar to Matlab code by google structure"""

  gnssRaw['Week'] = np.floor(-1*gnssRaw['FullBiasNanos']*1e-9/604800)
  gnssRaw['tRxNanos'] = (gnssRaw['TimeNanos']+gnssRaw['TimeOffsetNanos'])-(gnssRaw['FullBiasNanos']+gnssRaw['BiasNanos'])-gnssRaw['Week']*604800/1e-9
  gnssRaw['tRxSeconds'] = np.floor(gnssRaw['tRxNanos']/1e9)
  return gnssRaw, gnssAnalysis

def compute_pseudorange(gnssRaw, gnssAnalysis):
  """ Shubh wrote this similar to Matlab code by google structure"""
  gnssRaw['Pseudorange_ms'] = (gnssRaw['tRxNanos']-gnssRaw['ReceivedSvTimeNanos'])*1e-6
  gnssRaw['Pseudorange_meters'] = gnssRaw['Pseudorange_ms']*299792458*1e-3
  return gnssRaw, gnssAnalysis
