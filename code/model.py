import os
import dslib
from dslib.sandbox import treeano_trial_utils as ttu
from pprint import pprint
import numpy as np
sys.path.append('/home/rewon/dslib/experimental/projects/vascular_anomalies')
from vascular_libs import iou, negative_iou_differentiable
from vascular_libs import scale_to_window
import json
import dataset

trials_dir = os.path.join(os.environ['HOME'], '_trials_dir_')

params = dslib.AttrDict(

    # Model params
    # down_slabs=(1, 2, 3, 3, 3),
    # up_slabs=(2, 2, 2, 1, 1),
    down_slabs=(3, 3),
    up_slabs=(2, 2),
    data_downsample_factor=1,
    seg_downsample_factor=1,
    base_num_filters=64,
    dropout_probability=0.3,

    # training params
    num_chunks=10000,
    time_limit=4 * 60 * 60,
    learning_rate=2e-3,
    validate_interval=10,
    chunk_size=16,

    batch_shape=(16, 3, 300, 400)
    # data augmentation params
    # zoom_max_multiplier=1.6,
    # stretch_max_multiplier=1.3,
    # shear_max=0.01,
    # translation_max_distance=45,
    # rotation_max=0.0375,
    # horizontal_flip=0.5,
    # vertical_flip=0.0,
    # contrast_enhance_max=1.0,  # Range is [1/max, max] like with zoom
    # brightness_enhance_max=1.0,
    # affine_mode="constant",
)


def make_summary():
    from dslib.sandbox.summary import Summary

    def on_max_iou(old_val, new_val):
        print ">>> IOU Increased from %s to %s" % (old_val, new_val)
        ttu.save_network(trial, "best", network)
        trial.store_important("best_valid_iou", new_val)

    summary = Summary()
    summary.add("trial", value="%s:%d" % (trial_name, trial.iteration_num))
    summary.add("_iter", how="last")
    summary.add("_time", how="last")
    summary.add_recipe("s_per_iter")
    summary.add_recipe("ms_per_obs", params.chunk_size)
    summary.add_recipe("x max+iter",
                       "valid_iou",
                       on_change=on_max_iou,
                       format="%.4g")
    summary.add("final_iou",
                in_keys=["valid_iou"],
                how="last",
                format="%.4g")
    summary.add("final_cost_train",
                in_keys=["train_cost"],
                how="last",
                format="%.4g")
    summary.add("final_cost_valid",
                in_keys=["valid_cost"],
                how="last",
                format="%.4g")
    return summary


def make_model(params):
    import treeano.nodes as tn
    import treeano.sandbox.nodes.batch_normalization as bn
    import treeano.sandbox.nodes.input_scaling as input_scaling

    depth = len(params.down_slabs)

    layers = [
        tn.InputNode("x", shape=params.batch_shape),
        tn.DnnConv2DWithBiasNode("conv_1a", num_filters=32),
        tn.VeryLeakyReLUNode("relu_1a"),
        tn.GaussianDropoutNode("do_1a"),
        tn.DnnConv2DWithBiasNode("conv_1b", num_filters=64),
        tn.VeryLeakyReLUNode("relu_1b"),
        tn.GaussianDropoutNode("do_1b"),
        tn.DnnConv2DWithBiasNode("conv_1b", num_filters=64),
        tn.VeryLeakyReLUNode("relu_1b"),
        tn.GaussianDropoutNode("do_1b"),
        bn.BatchNormalizationNode("bn_1"),
        tn.MaxPool2DNode("mp_1", pool_size=(2, 2)),

        tn.DnnConv2DWithBiasNode("conv_2a", num_filters=64),
        tn.VeryLeakyReLUNode("relu_2a"),
        tn.GaussianDropoutNode("do_2a"),
        tn.DnnConv2DWithBiasNode("conv_2b", num_filters=128),
        tn.VeryLeakyReLUNode("relu_2b"),
        tn.GaussianDropoutNode("do_2b"),
        tn.DnnConv2DWithBiasNode("conv_2b", num_filters=128),
        tn.VeryLeakyReLUNode("relu_2b"),
        tn.GaussianDropoutNode("do_2b"),
        bn.BatchNormalizationNode("bn_2"),
        tn.MaxPool2DNode("mp_2", pool_size=(2, 2)),

        tn.DnnConv2DWithBiasNode("conv_3a", num_filters=128),
        tn.VeryLeakyReLUNode("relu_3a"),
        tn.GaussianDropoutNode("do_3a"),
        tn.DnnConv2DWithBiasNode("conv_3b", num_filters=256),
        tn.VeryLeakyReLUNode("relu_3b"),
        tn.GaussianDropoutNode("do_3b"),
        tn.DnnConv2DWithBiasNode("conv_3b", num_filters=256),
        tn.VeryLeakyReLUNode("relu_3b"),
        tn.GaussianDropoutNode("do_3b"),
        bn.BatchNormalizationNode("bn_3"),

        tn.SpatialRepeatNDNode('upsample1', upsample_factor=(2, 2)),
        tn.SpatialRepeatNDNode('upsample2', upsample_factor=(2, 2))
    ]

    layers.append(tn.DnnConv2DWithBiasNode(
        "final_conv",
        num_filters=1,
        filter_size=(1, 1)),
        )

    layers.append(tn.SigmoidNode('pred'))

    model = tn.HyperparameterNode(
        "model",
        tn.SequentialNode("model_seq", layers),
        inits=[treeano.inits.OrthogonalInit()],
        conv_pad='same',
        filter_size=(3, 3),
        dropout_probability=params.dropout_probability,
    )

    cost = negative_iou_differentiable

    model = tn.AdamNode(
        "adam", {
            "subtree": model,
            "cost": tn.TotalCostNode(
                "cost", {
                    "pred": tn.ReferenceNode("pred_ref", reference="pred"),
                    "target": tn.InputNode(
                        "y",
                        shape=(params.batch_shape),
                        dtype=theano.config.floatX
                    )
                },
                cost_function=cost
            )
        },
        learning_rate=params.learning_rate,
    )

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a model on whales.')
    parser.add_argument("-s", "--set-param", action="append",
                        help="Overrides a parameter. Use like '-s key=val'," +
                        " (can be specified multiple times)",
                        dest="param_overrides", default=[])
    args = parser.parse_args()

    trial_name = 'whale_segmentation'
    snippets = [
        ["data", "dataset.py"],
    ]

    with dslib.trial.run_trial(trial_name=trial_name,
                               trials_dir=trials_dir,
                               description=[],
                               snippets=snippets) as trial:
        os.mkdir(trial.file_path('segmentations'))

        # Store params for future reference.
        pprint(params)
        params_filepath = trial.file_path('params.json')
        with open(params_filepath, 'w') as f:
            json.dump(params, f)

        data = trial.load_module("data").Data(params)
        train_ds = data.dataset(is_validation_set=False)
        valid_ds = data.dataset(is_validation_set=True)

        with train_ds as train_gen, valid_ds as valid_gen:
            valid_chunk = valid_gen.next()

            # If we import theano after we open the datasets,
            # we can multiprocess
            import theano
            import treeano
            import canopy
            import canopy.sandbox.monitor_ui

            # Make Model/Network
            if params.previous_trial is not None:
                if params.use_previous_network:
                    print "Using network from %d" % params.previous_trial
                    network = ttu.load_previous_network(
                        trial_name=trial_name,
                        iteration_num=params.previous_trial,
                        trials_dir=trials_dir,
                        load_previous_architecture=True)
                else:
                    print "Using weights from %d" % params.previous_trial
                    model = make_model(params)
                    print(model)
                    network = model.network()
                    ttu.load_previous_network(
                        trial_name=trial_name,
                        iteration_num=params.previous_trial,
                        network=network,
                        trials_dir=trials_dir,
                        load_previous_architecture=False,
                        strict=False)

            else:
                print "Making new model."
                model = make_model(params)
                print(model)
                network = model.network()

            # build eagerly to share weights
            network.build()
            # to make sure network is serializable
            ttu.save_network(trial, "initial", network)

            # Validation
            valid_fn = canopy.handled_fn(
                network,
                [
                    canopy.handlers.time_call(key="valid_time"),
                    canopy.handlers.override_hyperparameters(
                        deterministic=True,
                        bn_use_moving_stats=False,
                        ),
                    canopy.handlers.evaluate_monitoring_variables(fmt="valid_%s"),
                    canopy.handlers.chunk_variables(batch_size=params.batch_size,
                                                    variables=["x", "y"])
                ],
                inputs={"x": "x", "y": "y"},
                outputs={
                    "valid_cost": "cost",
                    "valid_probs": "pred",
                    "window_min": ('windower', 'mins'),
                    "window_max": ('windower', 'maxs')
                    }
            )

            best_valid_iou = 0.0   # 0 is natural min.
            valid_iter = 0

            def add_validate_metrics(in_dict, result_dict):
                global best_valid_iou
                global valid_iter
                valid_iter += 1
                valid_out = valid_fn(valid_chunk)
                valid_out['window_max'] = valid_out['window_max'][0]
                valid_out['window_min'] = valid_out['window_min'][0]
                probs = valid_out.pop("valid_probs")
                preds = probs > 0.5
                result_dict.update(valid_out)
                valid_target = valid_chunk['y_full']
                this_iou = iou(preds, valid_target)
                result_dict["valid_iou"] = this_iou
                # Save an example image of the current best segmentation
                # Let's also save one as a snapshot
                if this_iou > best_valid_iou:
                    best_valid_iou = this_iou
                    store_image(
                        valid_chunk['x_full'][0, 0],
                        valid_chunk['y_full'][0, 0],
                        preds[0, 0],
                        filepath=trial.file_path('segmented_best.jpg'),
                        window_min=valid_out['window_min'],
                        window_max=valid_out['window_max'],
                        )
                    store_image(
                        valid_chunk['x_full'][0, 0],
                        valid_chunk['y_full'][0, 0],
                        preds[0, 0],
                        filepath=trial.file_path('segmentations/segmented_%04d.jpg' % valid_iter),
                        window_min=valid_out['window_min'],
                        window_max=valid_out['window_max'],
                        )

            def add_train_metrics(in_dict, result_dict):
                train_probs = result_dict.pop('train_probs')
                preds = train_probs > 0.5
                if not np.any(preds):
                    print ">>> Predicted 0 pixels."
                train_target = in_dict['y_full']
                result_dict["train_iou"] = iou(preds, train_target)

            # Training
            train_fn = canopy.handled_fn(
                network,
                [
                    canopy.handlers.time_call(key="total_time"),
                    canopy.handlers.call_after_every(params.validate_interval,
                                                     add_validate_metrics),
                    canopy.handlers.callback_with_input(add_train_metrics),
                    canopy.handlers.evaluate_monitoring_variables(fmt="train_%s"),
                    canopy.handlers.split_input(params.gpu_chunk_size, ["x", "y"]),
                    canopy.handlers.chunk_variables(batch_size=params.batch_size,
                                                    variables=["x", "y"]),
                ],
                inputs={"x": "x", "y": "y"},
                outputs={"train_cost": "cost", "train_probs": "pred"},
                include_updates=True
            )

            result_writer = canopy.sandbox.monitor_ui.ResultWriter(
                dirname=trial.file_path("monitor_ui"),
                # capture all variables
                pattern="",
                # keep matched keys
                remove_matched=False,
                symlink=True,
                default_settings_file='default_settings.json')

            # Summary
            summary = make_summary()

            # Reporting
            print_keys = {"_iter",
                          "_time",
                          "total_time",
                          "train_time",
                          "train_cost",
                          "train_iou",
                          "valid_time",
                          "valid_cost",
                          "valid_iou",
                          "window_min",
                          "window_max"
                          }

            def store_to_trial(train_log):
                trial.store("train_log", train_log, silent=True)
                if train_log["_iter"] % 10 == 0:
                    trial.store_important("iter", train_log["_iter"])
                summary.update(train_log)
                result_writer.write(train_log)
                pprint(dslib.toolz.keyfilter(lambda k: k in print_keys,
                                             train_log))

            # Run train loop
            canopy.evaluate_until(fn=train_fn,
                                  gen=train_gen,
                                  max_iters=params.num_chunks,
                                  max_seconds=params.time_limit,
                                  callback=store_to_trial)

            # Store summary
            summary_filepath = trial.file_path('summary.json')
            # value_dict has numpy values, need a numpy-aware serializer
            dslib.io_utils.json_dump(summary.to_value_dict(), summary_filepath)
