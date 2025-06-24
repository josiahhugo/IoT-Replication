# threading
import threading
from CIG_Feature_Selection import main as cig_main
from OpCode_graph import main as graph_main
from Eigenspace_Transformation import main as eigenspace_main
from export_data import main as export_main
# from adaboost_baseline import main as adaboost_main
#from junk_graphs import main as junk_main

def run_all():
    cig_main() # run feature selection first
    graph_main() # run graph building second
    eigenspace_main() # transformation
    export_main() # export data for MATLAB
    # adaboost_main() # adaboost baseline
    # junk_main()

if __name__ == "__main__":
    run_all()


