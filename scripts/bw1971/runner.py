import os
import time
import sys

os.environ['PATH'] = os.environ['PATH'] + ':/home/riley/gams/gams33.2_linux_x64_64_sfx'
# sys.path.append('/home/riley/gams/gams33.2_linux_x64_64_sfx')
# /home/riley/gams/gams33.2_linux_x64_64_sfx
#os.system('gams apr6_exp.gams resLim=7200 solver=BARON o=exp_baron.lst')
#time.sleep(7200)
#os.system('gams apr6_exp.gams resLim=300 solver=LINDO o=exp_lindo.lst')
#time.sleep(7200)
# os.system('gams apr6_exp.gams resLim=7200 solver=ANTIGONE o=exp_antigone.lst')
# time.sleep(7200)
#os.system('gams apr6_exp.gams resLim=7200 solver=SCIP o=exp_scip.lst')
#time.sleep(7200)
os.system('gams apr6_exp.gams resLim=7200 solver=ANTIGONE o=exp_antigone.lst')
