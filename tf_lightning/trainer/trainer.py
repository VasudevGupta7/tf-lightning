"""Trainer Class

@author: vasudevgupta
"""

class Trainer(object):
    
    def __init__(self, **kwargs):
        
        self.epochs= kwargs.get('epochs', 10)
        self.restore_ckpt= kwargs.get('restore_ckpt', False)    
        
    def fit(self, lightning_module, lightning_data_module):
        
        tr_dataset= lightning_data_module.train_dataloader()
        val_dataset= lightning_data_module.val_dataloader()
        
        lightning_module.train(tr_dataset, 
                               val_dataset,
                               epochs,
                               load_dir= None,
                               save_every_ckpt= False,
                               assert_consumed=False)

    def test(self):
        pass
    
    @classmethod
    def add_argparse_args(cls, parser):

        parser.add_argument('--epochs', type= int, default= 10, help= 'no of epochs')
        parser.add_argument('--restore_ckpt', type= int, default= 10, help= 'no of epochs')
        
        return parser
        
    @classmethod
    def from_argparse_args(cls, args):
        return cls(epochs= args.epochs, 
                   restore_ckpt= args.restore_ckpt)

