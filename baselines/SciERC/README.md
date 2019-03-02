# Training SciERC

* First you need to set up [SciERC](https://bitbucket.org/luanyi/scierc/src) repository.

* Script requirements
    * [SciSpaCy](https://allenai.github.io/scispacy/)

* For converting the data to the format accepted by SciERC you need to run:
    * `python prepare_input.py --input [the data json] 
    --output [path for saving scierc jsona file] 
    --output_whitespaces [path for saving temprorary file necessary for recovering after prediction]`    

* Then you need to train the model following the [SciERC](https://bitbucket.org/luanyi/scierc/src) instructions

* After training and predicting on a test file you can convert it back to our format using following command: 
    * `python recover.py --prediction [path to output jsona file]
    --scierc_input [path to the file that SciERC was called on (to obtain original sentences)]
    --whitespaces [the temprorary whitespaces file created by prepare_input.py]
    --output [path for saving the resulting json]`
