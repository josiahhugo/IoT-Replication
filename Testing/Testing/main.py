# threading
import threading
from CIG_Feature_Selection import main as cig_main
from OpCode_graph import main as graph_main
from Eigenspace_Transformation import main as eigenspace_main

def run_all():
    cig_main() # run feature selection first
    graph_main() # run graph building second
    eigenspace_main() # transformation

if __name__ == "__main__":
    run_all()


