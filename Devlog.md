# Thinking about the problem:

This implementation has a fairly simple cycle:
for number_of_games:  
    {Bidding -> reveal}

## Strategies
I can think of only 3 layers (heh) of strategies to learn:  

1. Bid on hands based on expected win rate
2. Bluff! 
3. Out-play your opponent

The first strategy is reasonable, if playing against a random opponent. 

The second strategy requires playing a 'smart' opponent, e.g. in self-play.

The third strategy would be ideal (if possible). If I can predict the opponent's move, I can out-play them.

## Modelling


### [ ✓ ] Mask Output state
We know some actions are illegal, and should not be chosen. There is an implementation that can be selected for choose_move (or in training?)

### Check Submission 
After step 1, I am unsure if choose move will successfully return an action for every state.  
It *should* be OK, as I am only seeing errors when the state is 'done' - which shouldnt call 'act' - but this needs to be checked.  
<span style="color:maroon">TODO</span>

### Better Understand the state of Play 
The current implementation gets better over time, as seen by average rewards over 100 episodes seeming to increase at towards ~0.4 / 0.5 

This may be the expected return when playing against a random opponent, but it is hard to understand intuitively. 

Some techniques that could help:
- Visualise Loss 
    - e.g. Using some more clever logic from [Tensorboard](https://www.tensorflow.org/tensorboard/get_started)

- Try and 'test' the bot in some known scenarios. 
    - e.g. Probability(raise) on pocket-aces should be high


### ?? Augment / Transform Input state
One opportunity is to compute directly the likelihood of winning for a hand, and feed that into the model. 
It would simplify the learning process significantly (since the input neuron dimension would just be 2 - win likelihood, and bet value). 
I think it's fair, based on some of the augmentation of state - e.g. [Mugit's spaceshooter](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a3294ba2-569c-4bd8-9f71-3e584be7ee07/team_honey_mugit.py?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220819%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220819T060314Z&X-Amz-Expires=86400&X-Amz-Signature=2f7dd8198bca067c691fecddea2c3765b6d20fb4430ec30590d256dd8d19a1b4&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22team_honey_mugit.py%22&x-id=GetObject)

`def augment(state):`  
`    # calculate angle between ships`  
`[...]` 

<span style="color:turquoise">TO REVIEW</span>



# Iterations on architectures:

## Step 1 - implement the very naive 
The Policy Gradient Algorithm from last time seems to 'learn' naively to play against a random opponent.

Iterations are much faster than in the previous game, so I tried running training over 100k episodes

To get this to run, I needed to implement the masking code.
I also am running across a number of issues with the game having no legal moves. This may be a problem in the actual submission - tbd!

![Returns over time, batch 1](Images/Step1_results.png)

## Step 2 - Sample behaviour on some expected scenarios
Pocket Aces should have a high probability to raise. Let's start with that!
