import os
import webbrowser

# log_file_dir = '//STEFFEN-PC/Sharing/FrontTransportReduction/NeuralFTR/training_results_local/NewTrainings/FullDec/Smoothing/'
log_file_dir = 'D:/Arbeit\FrontTransportReduction/NeuralFTR/training_results_local/FlamePinchOff/test2/'
webbrowser.open('http://localhost:6006')
os.system('tensorboard --logdir=' + log_file_dir)
