from model import run_pplm_example

Knob =  0.8
Affect = "joy"
Topic = "legal"
Prompt = "There was a"
result = run_pplm_example(
        affect_weight=1,  # it is the convergence rate of affect loss, don't change it :-p
        knob = Knob, # 0-1, play with it as much as you want
        cond_text=Prompt,
        num_samples=1,
        bag_of_words=Topic,
        bag_of_words_affect=Affect,
        length=50,
        stepsize=0.01,
        sample=True,
        num_iterations=3,
        window_length=5,
        gamma=1.5,
        gm_scale=0.95,
        kl_scale=0.01,
        verbosity='quiet'
    )

print(result)