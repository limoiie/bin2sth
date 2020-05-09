from ignite.engine import Events


def attach_stages(trainer, evaluator, ds_train, ds_val, ds_test):
    def eval_on(ds, tag, event):
        @trainer.on(event)
        def log_eval_results(engine):
            evaluator.run(ds)
            metrics = evaluator.state.metrics
            print("{} Results f Epoch: {}  "
                  "Avg accuracy: {:.2f} Avg loss: {:.2f}"
                  .format(tag, engine.state.epoch,
                          metrics['auc'], metrics['mse']))

    eval_on(ds_train, 'Training', Events.EPOCH_COMPLETED)
    eval_on(ds_val, 'Validation', Events.EPOCH_COMPLETED)
    eval_on(ds_test, 'Testing', Events.COMPLETED)


def attach_unsupervised_evaluator(trainer, evaluator, query_loader):
    @trainer.on(Events.EPOCH_COMPLETED)
    def eval_per_epoch(engine):
        evaluator.run(query_loader)
        metrics = evaluator.state.metrics
        print("Evaluation Results f Epoch: {}  "
              "AUC area: {:.2f}, Top-1 accuracy: {:.2f}"
              .format(engine.state.epoch,
                      metrics['auc'], metrics['topk-acc']))
