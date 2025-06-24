===NOTES FOR REPLICATION===

**WORK IN WSL**

**CORRECTED EXTRACTING OPCODES**:
    1) Install the ARM binutils:
        sudo apt-get update
        sudo apt-get install binutils-arm-linux-gnueabihf
        (to automatically check for valid ELF files)
        pip install python-magic
        sudo apt install libmagic1

    2) Correct script (for individual file):
        arm-linux-gnueabi-objdump -d /home/josiah/research/IoTMalwareDetection-master/Benign/all_goodware/combined1/3Dc
    3) Run extract_opcodes.py


Creating a venv:
    1) sudo apt install python3.12-venv
    2) python3 -m venv venv
    3) source venv/bin/activate

Creating proress bar:
    1) pip install alive-progress

Feature Selection using Class-wise Information Gain:
    1) install libraries:
        pip install numpy
        pip install pandas
        pip install scikit-learn
   
    2) Reading Benign Opcode files:
        = Go through combined1-3 folders where each .opcode file (disassembled binary) is read
        and treated as a single text string of opcodes    
        = Each string is added to the samples list, and 0 is added to labels to mark it as benign
    
    3) Reading malware Opcode files
        = Similar process to benign but malware

    4) Total Summary:
        = 1207 samples in total --> each sample is a sequence of opcodes

    5) N-Gram Vectorization:
        = CountVectorizer to convert opcode sequences into numerical vectors
        = N-gram range (1, 2) --> captures individual opcodes (1-grams) and pairs of opcodes (2-grams)
        = max_features=10000 --> restricts to the 10,000 mostt frequent opcode patters to avoid memory issues
            = 5000 for fast testing | 10000 for balance | 50000 if your machine is strong
        = Output: 1207x1000 matrix (x) where:
            = rows is samples
            = columns is opcode n-grams
            = values is counts of those n-grams

    6) CIG Calculation:
        = computes how much informative each feature (n-gram) is dor distiguishing between benign and malware
        = CIG is computted per feature with respect to both classes
        = Formula combines information gain for each class --> rewards features that are:
            = common in one class
            = rare in the other

    7) Feature selection:
        = Combines benign and malware CIG scores per feature
        = Selects top 82 features with the highest combined scores
        = Reduces the original 10,000-dimension matrix to 82 dimensions

    8) Final output:
        = New X_selected matrix has 1207 samples & only the 82 most informative n-gram features
        = Sample output: topopcode patterns... likely hexadecimal instruction encodings

Build OpCode Graphs:
    Notes to consider:
        Nodes = selected opcode n-grams (82 features)
        Edges = transitions between opcodes in that sample (based on their order in the opcode sequence)
        Weights = frequency of trasitions between features
    
    2) Build feature index map
        = map each selected feature (n-grams) to its corresponding index in the 82-feature list
    
    3) Tokenize each sample and extract valid n-grams
        = each sample (string of opcodes) must be turned into 1 and 2-grams, only keep the ones in your 82-feature list
    
    4) Construct 82x82 adjacency matrix for each sample
        = for each opcode sequence:
            = extract relevant n-grams (in order)
            = for each pair of consecutive n-grams that both exist in the top 82 features:
                = add an edge in the adjacency matric (increment weight)
    
    5) generate graphs for all samples

    6) OUTPUT:
        = adj_matrices: A list of adjacency matrices (one graph per sample)
        = these graphs encode opcode trainsition pattersn using only the top 82 n-gram features
    
    **RUN VISUALIZE_GRAPH SEPARATELY**

    Visualizing the graph:
        1) install libraries:
            pip install networkx
            pip install matplotlib
        
        2) load the adjacency matrices

        3) load the selected feature names

        4) choose sample index to visualize

        5) try finding a valid graph

        6) create directed graph

        7) add edges with weights from the adjacency matrix

        8) plot

        Understanding the graph:
            = Nodes --> opcode n-gram 
                = among the top 82 features selected using Class-Wise Information Gain
                = labeled in hexadecimal representations of opcodes (n-gram hashes)
            = edges
                = node A -> node B: disassembled opcode sequence A was followed by Benign
                = transitions form the behavioral structure of the program
            = weights
                = the normalized, distance-weighted likelihood that one opcode n-gram follows another within that specific sample
            = structure shape 
                = reveals execution flow copmlexity which is useful for distinguishing files

    
Eigenspace transformation:
    1) compute top k eigenvalues and eigenvectors
    
    2) Flatten top-k eigenvectors into a single vector
    
    3) Vector becomes the feature representation of that sample for training a classifier

    **paper uses k = 12**

    Understanding the graph:
        = each dot is a binary sample
            = sample was originally a graph
            = ran eigenspace transformation --> extracted top 12 eigenvectors (984D vector)
            = PCA to reduce those 984 dimensions to 2 for visualization
        = component 1 & 2 --> abstract dimensions that best separate the variance in your data
            = don't directly correspond to any specfici opcode or instruction, just directions in the high-dimensional feature space where the samples differ most
        = clusters & spread
            = tight clusters likely correspond to similar behavior --> maybe benign samples
            = outliers may be malware with distincly different opcode transition patterns
            = visually separated groups suggest real behavioral differences in the underlying opcode graph structure

CNN with eigenspace embeddings:
    1) install libraries:
        pip install torch
    = traing a CNN on eigenspace embeddings (X_graph) to classify samples as malware(1) or benign(0)
    TESTING:
        = Benign has high accuracy --> overfitting
            = malware --> never predicted any malware

CNN Training Using MATLAB:
    1) download MATLAB
    2) create matlab file but outside of WSL
    3) ISSUE:
        = imbalanced datasets: more benign than malware
            = fix 1: subsample (128 | 128)
                = 80/20 split --> better
                    = more training data: sees more benign/malware examples to learn from 
                    = larger test set: better estimate of generalization performance
                    = no extra validation set: avoids overfitting to a tiny validation set

                = 70/10/20 split --> not good
                    = limited number of malware samples
                    = benign data is relatively more abundant and diverse
                    = validation set was too small to be reliable
        = Possibilities to why results are lower than paper:
            = sample size --> I am currently doing 128;128 due to there being more benign in the dataset
            = preprocessing quality --> paper opcode files may be better normalized
            = CNN architecture --> paper may have more layers (I amd currently using 3 layers), batch normalization, L2 regularization, learning rate decay


    IMPORTANT TERMINOLOGY
        Epoch: number of times the model will iterate over the entire training dataset during training
            analogy: how many times you review the full study guide
        Batch size: number of training samples the model uses in one forward and backward pass
            analogy: how many flashcards you quiz yourself on at a time
        Class weights: assign different importance (penalties) to each class when the model makes an error
        K-fold Cross-Validation: evaluate the performance of a model by partitioning the dataset into k subsets (folds)



**SUMMARY**
= Opcodes known for being efficient for malware detection
    = extract opcodes to get n-gram opcode sequence which is a common approach to classify malware based on their disassembled codes
= Use Class-Wise Information Gain (CIG) to overcome global feature selection imperfection and recognize more useful features
    = selects the features that best distinguish between benign and malicious samples
= OpCode graph captures contextual relationships between OpCodes
= Eigenspace transformation (graph embedding) --> transform each opcode graph into a fixed-size vector
    = convert variable and complex graph structures into standard input for machine learning models
