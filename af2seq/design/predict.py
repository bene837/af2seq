import os
try:
    from pyrosetta import *
except ImportError:
    print('Warning: Couldnt find pyrosetta. Some features will not work properly')

from af2seq.alphafold.notebooks import notebook_utils
from af2seq import MCMCDesign
# Load correct tqdm progress bar for terminal or jupyter notebook
try:
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except:
    from tqdm import tqdm


import py3Dmol
import time
import numpy as np
import jax

from af2seq.alphafold.common import protein
from af2seq.alphafold.data import pipeline_multimer
from ..alphafold.data import parsers, feature_processing, msa_pairing, pipeline, templates
from ..alphafold.model import model as pred
from ..alphafold.common import confidence
from .pre_processing import build_msa, build_msa_ptm, build_pssm, load_msa


import warnings
warnings.filterwarnings("ignore")

class ProteinPredict(MCMCDesign):
    def __init__(self, output_path: str = None):
        super(ProteinPredict, self).__init__(0, output_path, False)
        self.model_loaded = False
        self.compiled_model = None

    def plot_pred(self, nr: int = 0):
        p = py3Dmol.view(js="https://3dmol.org/build/3Dmol.js")
        result0 = jax.tree_map(lambda x: x[nr], self.best_result["result"])
        if self.modeltype == "ptm":
            p.addModel(
                protein.to_pdb(
                    protein.from_prediction(self.best_result["feat"], result0)
                ),
                "pdb",
            )
        else:
            p.addModel(
                protein.to_pdb(
                    protein.from_prediction(
                        self.best_result["feat"],
                        result0,
                        remove_leading_feature_dimension=False,
                    )
                ),
                "pdb",
            )
        p.setStyle({"cartoon": {"color": "spectrum"}})
        p.zoomTo()
        return p.show()

    def save(self, name, nr: int = 0):
        result0 = jax.tree_map(lambda x: x[nr], self.best_result["result"])
        if self.modeltype == "ptm":
            pdb = protein.to_pdb(
                protein.from_prediction(self.best_result["feat"], result0)
            )
        else:
            pdb = protein.to_pdb(
                protein.from_prediction(
                    self.best_result["feat"],
                    result0,
                    remove_leading_feature_dimension=False,
                )
            )

        with open(name, "w") as f:
            f.write(pdb)

    def feat_update(
            self,
            sequences,
            msas: list = None,
    ):
        model_type_to_use = self.modeltype
        features_for_chain = {}
        raw_msa_results_for_sequence = {}

        for sequence_index, (sequence, msa) in enumerate(zip(sequences, msas), start=1):

            if msa == None:
                # print(sequence)
                msa = parsers.Msa([sequence], [[0] * len(sequence)], ["query"])

            single_chain_msas = [msa]
            uniprot_msa = msa

            # Turn the raw data into model features.
            feature_dict = {}
            feature_dict.update(
                pipeline.make_sequence_features(
                    sequence=sequence, description="query", num_res=len(sequence)
                )
            )
            feature_dict.update(pipeline.make_msa_features(msas=single_chain_msas))

            # Force input template
            if self.temp_paths != None and self.modeltype == "ptm":
                # Get inputs
                name = self.temp_paths.split('/')[-1]
                template_path = '/'.join(self.temp_paths.split('/')[:-1])

                # Create template features
                template_featurizer = templates.HhsearchHitFeaturizer(
                    mmcif_dir=template_path,
                    max_template_date="2100-01-01",
                    max_hits=20,
                    kalign_binary_path="kalign",
                    release_dates_path=None,
                    obsolete_pdbs_path=None)

                index = 1
                aligned_cols = len(sequences[0])
                sum_probs = 100.0
                query = sequences[0]
                hit_sequence = sequences[0]
                indices_query = range(aligned_cols)
                indices_hit = range(aligned_cols)

                hit = parsers.TemplateHit(index, name, aligned_cols, sum_probs, query,
                                          hit_sequence, indices_query, indices_hit)

                temps = template_featurizer.get_templates(sequences[0], [hit])
                temps = temps.features
                feature_dict.update(temps)

            else:
                # Add only empty placeholder features.
                feature_dict.update(
                    notebook_utils.empty_placeholder_template_features(
                        num_templates=0, num_res=len(sequence)
                    )
                )

            # Construct the all_seq features only for heteromers, not homomers.
            if len(set(sequences)) > 1:
                valid_feats = msa_pairing.MSA_FEATURES + (
                    "msa_uniprot_accession_identifiers",
                    "msa_species_identifiers",
                )
                all_seq_features = {
                    f"{k}_all_seq": v
                    for k, v in pipeline.make_msa_features([uniprot_msa]).items()
                    if k in valid_feats
                }
                feature_dict.update(all_seq_features)

            features_for_chain[protein.PDB_CHAIN_IDS[sequence_index - 1]] = feature_dict

        if model_type_to_use == "ptm":
            np_example = features_for_chain[protein.PDB_CHAIN_IDS[0]]

        elif model_type_to_use == "multimer":
            all_chain_features = {}
            for chain_id, chain_features in features_for_chain.items():
                all_chain_features[
                    chain_id
                ] = pipeline_multimer.convert_monomer_features(chain_features, chain_id)

            all_chain_features = pipeline_multimer.add_assembly_features(
                all_chain_features
            )

            np_example = feature_processing.pair_and_merge(
                all_chain_features=all_chain_features, is_prokaryote=False
            )

            # Pad MSA to avoid zero-sized extra_msa.
            np_example = pipeline_multimer.pad_msa(np_example, min_num_seq=512)

            np_example["target_feat"] = jax.nn.one_hot(np_example["aatype"], 21)

        np_example = self.model.process_features(np_example, random_seed=0)
        return np_example

    def design(self):
        raise NotImplemented

    def _configure_model(self, modeltype, recycles, msa_paths):
        (
            self.model_config,
            single_params,
            self.model_params,
        ) = self._select_and_load_params(modeltype, recycles)
        self.model = pred.RunModel(self.model_config, single_params)

    def _prep_model(self):
        # AF FUNCTION
        def structure_change_fn(processed_feature_dict, model_params):
            result = self.model.apply(
                model_params, jax.random.PRNGKey(0), processed_feature_dict
            )

            return result

        alphafold_fwd = jax.jit(jax.vmap(structure_change_fn, in_axes=(None, 0)))
        return alphafold_fwd

    def generate_ptm_msa(self, msa_paths, seq_lengths):
        # find out the msa depth
        max_depths = []
        for i in msa_paths:
            if i != None:
                _, msa = load_msa(i)
                max_depths.append(len(msa))

        max_msa_depth = min(max_depths)

        new_msa = [""] * max_msa_depth

        for i in range(len(msa_paths)):
            msa_path = msa_paths[i]
            if msa_path == None:
                # go through the max_msa_depth
                for j in range(1, max_msa_depth, 2):
                    new_msa[j] = new_msa[j] + "-" * seq_lengths[i]
            else:
                _, msa = load_msa(msa_path)
                for j in range(max_msa_depth):
                    if not msa[j].startswith(">"):
                        new_msa[j] = new_msa[j] + msa[j]
                    else:
                        new_msa[j] = msa[j]

        for i in range(1, max_msa_depth, 2):
            new_msa[i] = new_msa[i] + "\n"

        self.new_msa = new_msa

        a3m = "".join(new_msa)
        new_msa = parsers.parse_a3m(a3m)

        return new_msa, seq_lengths

    def predict(
            self,
            seq: list = None,
            msa_paths: list = None,
            temp_paths: list = None,
            filename: str = None,
            recycles: int = 3,
            modeltype: str = None,
    ):

        if self.start_time is None:
            self.start_time = time.asctime()

        self.startseq = seq
        self.name = filename
        self.temp_paths = temp_paths

        if msa_paths == None:
            self.use_msas = False
        else:
            self.use_msas = True

        if isinstance(seq, str):
            seq = [seq]

        # BUILD MODEL
        if modeltype is None:
            self.log.debug(f"chains {len(seq)}")
            modeltype = "ptm" if len(seq) == 1 else "multimer"
        self.modeltype = modeltype

        self.log.info(f"Using {self.modeltype} model")

        if self.model_loaded == False:
            self._configure_model(self.modeltype, recycles, msa_paths)
            self.compiled_model = self._prep_model()
            self.model_loaded = True

        sequences = seq

        if modeltype == "ptm" and len(seq) > 1:
            # determine the lengths of the chains
            seq_lengths = []
            for i in sequences:
                seq_lengths.append(len(i))

        if modeltype == "ptm" and msa_paths != None:
            msa = build_msa_ptm(msa_paths, seq_lengths, sequences)
            self.msas = [msa]
            sequences = ["".join(sequences)]
        elif modeltype == "ptm" and msa_paths != None:
            self.msas = [None]
            sequences = ["".join(seq)]
        else:
            msas = []
            for msa_path in msa_paths:
                msa, _ = load_msa(msa_path)
                msas.append(msa)
            self.msas = msas

        # Here an empty template is generated!
        feat = self.feat_update(sequences, self.msas)

        # copy atoms to masks. Masks arent used, this prevents crashing in prediction mode.
        if modeltype == "ptm":
            feat["positional_mask"] = feat["atom37_atom_exists"].copy()
            feat["sidechain_mask"] = feat["atom37_atom_exists"].copy()
        else:
            feat["positional_mask"] = feat["all_atom_mask"].copy()
            feat["sidechain_mask"] = feat["all_atom_mask"].copy()

        if modeltype == "ptm" and len(seq) > 1:
            # Determine the chain breaks
            chain_break = []
            for i in range(len(seq_lengths) - 1):
                if i == 0:
                    chain_break.append(seq_lengths[i])
                else:
                    chain_break.append(sum(seq_lengths[: i + 1]))

            residue_index = np.copy(feat["residue_index"])
            for j in chain_break:
                for i in feat["residue_index"][0]:
                    if j <= i:
                        residue_index[0][i] += 200

            feat["residue_index"] = residue_index

        # for debugging purposes
        self.feat = feat

        self.model.init_params(feat)

        result = self.compiled_model(feat, self.model_params)
        if self.modeltype == "multimer":
            result, _ = result

        result0 = jax.tree_map(lambda x: x[0], result)

        if self.modeltype == "ptm":
            self.pdbs.append(protein.to_pdb(protein.from_prediction(feat, result0)))
        else:
            self.pdbs.append(
                protein.to_pdb(
                    protein.from_prediction(
                        feat, result0, remove_leading_feature_dimension=False
                    )
                )
            )

        self.best_result["feat"] = feat
        self.best_result["result"] = result

        if self.output_path is not None:
            fname = (
                self.name
                if self.name is not None
                else f"design_{self.target_file.split('/')[-1].split('.pdb')[0]}"
            )
            out_file = f"{fname}_{self.ident}.json"
            file = os.path.join(self.output_path, out_file)

            self.best_result["TM_score"] = 0  # tm
            self.best_result["RMSD"] = 0  # rmsd
            print(f"Saving to {file}")
            self.generate_report_and_pdb(file)

    def get_confidence_metrics(self, nr: int = 0):
        """Post processes prediction_result to get confidence metrics."""

        prediction_result = jax.tree_map(lambda x: x[nr], self.best_result["result"])
        multimer_mode = self.modeltype == "multimer"

        confidence_metrics = {}
        confidence_metrics["plddt"] = confidence.compute_plddt(
            prediction_result["predicted_lddt"]["logits"]
        )
        if "predicted_aligned_error" in prediction_result:
            confidence_metrics.update(
                confidence.compute_predicted_aligned_error(
                    logits=prediction_result["predicted_aligned_error"]["logits"],
                    breaks=prediction_result["predicted_aligned_error"]["breaks"],
                )
            )
            confidence_metrics["ptm"] = confidence.predicted_tm_score(
                logits=prediction_result["predicted_aligned_error"]["logits"],
                breaks=prediction_result["predicted_aligned_error"]["breaks"],
                asym_id=None,
            )
            if multimer_mode:
                # Compute the ipTM only for the multimer model.
                confidence_metrics["iptm"] = confidence.predicted_tm_score(
                    logits=prediction_result["predicted_aligned_error"]["logits"],
                    breaks=prediction_result["predicted_aligned_error"]["breaks"],
                    asym_id=prediction_result["predicted_aligned_error"]["asym_id"],
                    interface=True,
                )
                confidence_metrics["ranking_confidence"] = (
                        0.8 * confidence_metrics["iptm"] + 0.2 * confidence_metrics["ptm"]
                )

        if not multimer_mode:
            # Monomer models use mean pLDDT for model ranking.
            confidence_metrics["ranking_confidence"] = np.mean(
                confidence_metrics["plddt"]
            )

        return confidence_metrics