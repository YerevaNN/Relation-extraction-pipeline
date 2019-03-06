# Training SciERC

* First you need to set up [SciERC](https://bitbucket.org/luanyi/scierc/src) repository.

* Script requirements
    * [SciSpaCy](https://allenai.github.io/scispacy/)

* For converting the data to the format accepted by SciERC you need to run:
    * `python prepare_input.py --input [the data json] 
    --output [path for saving scierc jsona file] 
    --output_whitespaces [path for saving temprorary file necessary for recovering after prediction]`    

* Then you need to train the model following the [SciERC](https://bitbucket.org/luanyi/scierc/src) instructions
    * First, we need to generate ELMo vectors using `generate_elmo.py`. 
    Currently, the filenames are hardcoded in the script. 
    We have to edit the file for each run. The script produces huge HDF5 files, 
    558 / 76 / 162 MB for train/dev/test sets.
    * Then we need to edit `experiments.conf`. 
    * In `glove_300d_filtered` and `glove_300d_2w` set paths to `GloVe` files
    * In `scientific_entity` part we use:
    
          ner_labels = ["tissue", "experimental_construct", "brand", "protein_isoform", "recombinant_protein", "protein", "gene", "fusion_protein", "RNA_family", "experiment_tag", "peptide", "parameter", "chemical", "DNA", "protein_family", "cell", "gene_family", "protein_DNA_complex", "protein_motif", "protein_region", "disease", "other", "RNA", "amino_acid", "fusion_gene", "organism", "assay", "protein_RNA_complex", "protein_domain", "organelle", "drug", "protein_complex", "reagent", "process", "mutation"]
          relation_labels = ["bind"]
    
    * Then we add the following paths in `scientific_elmo`:

          lm_path = "../1.0alpha4.train.scierc.elmo.hdf5"
          lm_path_dev = "../1.0alpha4.dev.scierc.elmo.hdf5"
          train_path = "../1.0alpha4.train.scierc.json"
          eval_path = "../1.0alpha4.dev.scierc.json"
          
    * Finally, we make sure NER is also being trained:
    
          scientific_n0.1c0.3r1 = ${scientific_best_relation} {
            ner_weight = 0.1
            output_path = "../1.0alpha4.dev.scierc.output.n0.1c0.3r1.json"
          }
    
    * Run `./scripts/build_custom_kernels.sh`
    
    * If the environment is Python 3, `2to3` tool should be used to 
    convert all files to Python 3: `2to3 -w *.py`. 
    Additionally, `utils.py` has a line which should be converted manually:
    
        `c.encode("utf-8").strip()` on line 42.
        
        Also, `srl_eval_utils.py` should have two changes: 
        
        `num_predictions = (k * text_length) // 100`
        
        `print("Predicted", list(zip(sorted_starts, sorted_ends, sorted_scores))[:len(gold_spans)])`
    
    * Run `python singleton.py scientific_n0.1c0.3r1` for training.
    
    * In parallel, run `python evaluator.py scientific_n0.1c0.3r1` for evaluating during training 
    (the main script does *not* do evaluation)
    
    * We get the best results at around 2000 iterations
    
    * The following command will produce an output on the dev set:
    
          python write_single.py scientific_n0.1c0.3r1


* After training and predicting on a test file you can convert it back to our format using following command: 
    * `python recover.py --prediction 1.0alpha4.dev.scierc.output.n0.1c0.3r1.json
    --scierc_input 1.0alpha4.dev.scierc.json
    --whitespaces 1.0alpha4.dev.scierc.whitespace
    --output 1.0alpha4.dev.prediction.scierc_n0.1c0.3r1.json`
