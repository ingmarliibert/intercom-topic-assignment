import subprocess
import gc

n_topics = [3, 5, 7]
learning_decays = [.5, .7, .9]

for n_topic in n_topics:
    for learning_decay in learning_decays:
        if n_topic == 3 and abs(learning_decay - 0.5) < 0.1:
            continue
        try:
            print(n_topic, learning_decay)
            args = ["python", "train_model_w_args.py", str(n_topic), str(learning_decay)]
            out = subprocess.run(args, stdout=subprocess.PIPE)
            gc.collect()
            print(out.stdout)
        except Exception as e:
            print(e)
            continue