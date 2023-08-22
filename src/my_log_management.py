import hydra
from omegaconf import DictConfig
import shutil
import yaml
import time
import pytz
from datetime import datetime

class logging_manager():
    
    def __init__(self, config_filename, logfile_path):
        self.config_filename = config_filename
        self.logfile_path = logfile_path    
        
    def strt_time(self):
        timezone = pytz.timezone('Europe/Berlin')
        start_time = datetime.now(tz=timezone)
        self.start_time = start_time
        
        return start_time.strftime("%a%H:%M:%S")
    
    def end_time(self):
        timezone = pytz.timezone('Europe/Berlin')
        end_time = datetime.now(tz=timezone)
        self.ending_time = end_time
        
        return end_time.strftime("%a%H:%M:%S")
    
    def log(self, test_result):
                
        with open('03-main_config.yaml', 'r') as file:
            config_data = yaml.load(file, Loader=yaml.FullLoader)
            
        with open(self.logfile_path, 'a') as f:
            #start time
            f.write(self.start_time.strftime("%a%H:%M:%S"))
            f.write("\n")
            
            #training config
            yaml.dump(config_data["model"], f)
            
            #test results
            f.write("loss\taccuracy\tf1_score")
            f.write("\n")
            test_result = [str(r) for r in test_result]
            f.write("\t".join(test_result))
            f.write("\n")
            
            #run time
            runtime = self.ending_time - self.start_time
            f.write(str(runtime))
            f.write("\n")
            
            # end time
            f.write(self.ending_time.strftime("%a%H:%M:%S"))
            f.write("\n")
            f.write("\n")
        
    def change_config(self, new_SEQ_LEN = 50, new_epochs = 30, new_batch_size = 32, new_learning_rate = 0.01):
        config_filename = "{}.yaml".format(self.config_filename)
        
        # open, change an write yaml file
        with open(config_filename, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            
            data["model"]['SEQ_LEN'] = new_SEQ_LEN
            data["model"]['epochs'] = new_epochs
            data["model"]['batch_size'] = new_batch_size
            data["model"]['learning_rate'] = new_learning_rate
            
            with open(config_filename, 'w') as file:
                yaml.dump(data, file)

