import json
from src.utils.tune import get_ranksim_hyparam
from src.data.loader import TripletDataIterator
from src.model.sort.ranksim import RankSimSort
from src.layers.augment import AugmentModel
from src.utils.experiments import augment_config, setup, get_callbacks
from tensorflow.keras.optimizers import Adam

def train(args):
    hyparams = json.load(open(args.params))
    aug_config = augment_config(hyparams['augment'])
    exp_dir = setup(args, hyparams)
    # Open a json file containing the dataset information
    # using the dataset name and data_dir from args
    with open(args.data_dir + "/" + args.dataset + "/sort.annotations.json", 'r') as f:
        dataset = json.load(f)

    data_iter = TripletDataIterator(dataset['train'], 
                                    args.data_dir, 
                                    hyparams["input_shape"]).build(hyparams["batch_size"], True, True)
    val_iter = TripletDataIterator(dataset['val'], 
                                    args.data_dir, 
                                    hyparams["input_shape"]).build(1, False, False)

    # Load the model
    augmentor = AugmentModel(**aug_config)
    model = RankSimSort(hyparams["input_shape"], hyparams["gamma"], hyparams["weights"], augmentor)

    optimizer = Adam(learning_rate=hyparams['lr'])
    model.compile(optimizer, hyparams['f_lambda'])

    # Callbacks
    callbacks = get_callbacks(exp_dir, 'val_sort_loss')

    # Train the model
    model.fit(data_iter, epochs=hyparams['epochs'], callbacks=callbacks, 
            validation_data=val_iter, validation_freq=1)