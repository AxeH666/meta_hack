# allinone

## DETAILED_REQUIREMENTS.docx

Functional Requirements
Real-world task simulation
The environment must simulate a task humans actually do. Not games, not toys. Examples: email triage, code review, data cleaning, scheduling, customer support, content moderation.
OpenEnv spec compliance
Implement the full OpenEnv interface: typed Observation, Action, and Reward Pydantic models. step(action) → returns observation, reward, done, info. reset() → returns initial observation. state() → returns current state. openenv.yaml with metadata. Tested via openenv validate.
Minimum 3 tasks with agent graders
Each task defines a concrete objective an agent must accomplish, with a programmatic grader that scores performance (0.0–1.0). Tasks should range: easy → medium → hard. Graders must have clear, deterministic success/failure criteria.
Meaningful reward function
Provides signal over the full trajectory (not just binary end-of-episode). Rewards partial progress toward task completion. Penalizes clearly undesirable behavior (e.g. infinite loops, destructive actions).
Baseline inference script
Uses the OpenAI API client to run a model against the environment. Reads API credentials from environment variables (OPENAI_API_KEY). Produces a reproducible baseline score on all 3 tasks.
Detailed Requirements
Non-Functional Requirements
Deploys to a Hugging Face Space
Environment must run as a containerized HF Space tagged with openenv.
Containerized execution
Must include a working Dockerfile. The environment should start cleanly with docker build + docker run.
Documentation
README must include: environment description and motivation, action and observation space definitions, task descriptions with expected difficulty, setup and usage instructions, baseline scores.


## evaluation criteria.docx

Parameter
Weight
Description
Real-world utility
30%
Does the environment model a genuine task? Would someone actually use this to train or evaluate agents?
Task & grader quality
25%
Are tasks well-defined with clear objectives? Do graders accurately and fairly measure success? Meaningful difficulty progression?
Environment design
20%
Clean state management, sensible action/observation spaces, good reward shaping, proper episode boundaries.
Code quality & spec compliance
15%
Follows OpenEnv spec, clean project structure, typed models, documented, tested, Dockerfile works.
Creativity & novelty
10%
Novel problem domain, interesting mechanics, clever reward design, original approach.
Scoring Breakdown
Real-world utility (30%)
• 0–5: Toy/artificial problem with no practical application
• 6–15: Valid domain but shallow modeling of the real task
• 16–25: Good domain modeling, would be useful for agent evaluation
• 26–30: Excellent — fills a real gap, immediate value for the RL/agent community
Task & grader quality (25%)
• 3+ tasks with difficulty range?
• Graders produce scores between 0.0–1.0?
• Graders deterministic and reproducible?
• Hard task genuinely challenges frontier models?
Environment design (20%)
• reset() produces clean state?
• Action/observation types well-designed and documented?
• Reward function provides useful varying signal (not just sparse)?
• Episode boundaries sensible?
Code quality & spec compliance (15%)
• openenv validate passes?
• docker build && docker run works?
• HF Space deploys and responds?
• Baseline script runs and reproduces scores?
Creativity & novelty (10%)
• Domain we haven’t seen in OpenEnv before?
• Reward design has interesting properties?
• Clever mechanics that make the environment engaging?


## HOW_judging_works.docx

Phase 1: Automated Validation
Pass/fail gate — HF Space deploys, OpenEnv spec compliance, Dockerfile builds, baseline reproduces, 3+ tasks with graders.
Phase 2: Agentic Evaluation
Scored — baseline agent re-run, standard Open LLM agent (e.g. Nemotron 3 Super) run against all environments, score variance check.
Phase 3: Human Review
Top submissions reviewed by Meta and Hugging Face engineers for real-world utility, creativity, and exploit checks.
Disqualification Criteria
Environment does not deploy or respond
Plagiarized or trivially modified existing environments
Graders that always return the same score
No baseline inference script


## meta.txt

Hello everyone. A very warm welcome to all of you. Hi. So, we'll wait for a couple of more minutes before we get started, right? So that more people can join and uh once we have a once we have more people with us, we'll get started with the session. All right? I'll be talking to you guys soon again. Hey everyone, a very warm evening to all of you. Hope you all are doing great. Uh, welcome to the session and uh, please acknowledge in the chat if you guys are able to hear me and see me.

Am I audible and visible? All right guys, so you know just to warm up I would like you all to you know just quickly tell me uh you all have of course registered for the hackathon. So just share your team name and you know uh probably which city are you hailing from in case you guys are from multiple cities just the leader city is fine for now but yes uh you know we're just uh getting a sense of if the chat is all good and uh we are ready to set rolling. Yeah. Perfect. Great. All right guys, so thank

you so much for engaging in the chat and uh good to see all the excitement. So we'll begin now and uh yeah so firstly I would quickly uh like to set some context. Um you know you all have registered for the for the hackathon and uh the idea for this particular session is to get you guys started quickly right we do have a lot of different tools right you know already that uh this hackathon needs to be done using what we have what we call as open envir so to get you guys quickly started especially

if you've been struggling uh to you know get started and uh make it easy for you to perform step number one for the hackathon. We are setting up this particular session and uh with me today I have uh Ben who's from hugging face and uh Ben will be talking uh with you guys very soon and uh he will be giving you guys a walk through of what reinforcement learning is. We are going to keep that very short, right? He'll set context for how the library open ENB makes it very easy and straightforward

for you to develop your own environments, which is exactly what you need to do for this particular hackathon. Right. So over to you Ben and uh let's quickly uh welcome Ben and yes over to you. >> Thanks Pit. Yes, so I'm looking forward to giving this talk. Um and basically the talk is really about how to build a real world M. So like really just how to win the hackathon could be the the simplest way of putting it. Uh and when I whenever anyone asks me this my answer is pretty straightforward. It's that I

think a good environment is one that I could imagine one of my post-training or reinforcement learning colleagues using in one of their training runs. So would they put this into their training run and use it because it represented a real world task? So if you take a product like Claude codeex an environment like Git would be a perfect environment to put into that product. Now, we already have environments for Git, so that's not something that you could win this hackathon with. But if you think about

future products, maybe ones that uh operate over calendars and kind of uh email triage or or other kind of use cases that we haven't really thought of yet, let's say in specific domains like healthcare, the environments that those products would need. So let's say if you had a kind of clawed code for health, then what environment would that product need? That that's the real takeaway of like what a good MV is. Okay, something that really fits with a product and and a use case. So to understand that, we have to first

understand what RL is. So reinforcement learning. So we're going to go over a brief overview of reinforcement learning. And then uh we're going to go into the environment itself. So if you don't know the the project openm sometimes called open env but it but it's really open m it's a short for environment. It is a collaboration between meta torch plugging face and unsloth. Because of that, I have some really nice uns slides from the Unsloth team and they're going to walk us

through reinforcement learning. So, we're going to start off with just intuition, right? Let's say that we ask an LLM to write a fast matrix multiplication function. It could be any Python function. It could be a whole codebase, but in this example, we're just saying generate a fast matrix multiplication function. We give it some formatting constraints. In this case, put the code between triple back ticks. The model will produce an implementation. But this is just a single generation.

What happens when we want to generate many of these? We can't um call the model once. We call it many times to get different candidate implementations. Then we feed each one of those implementations to an RL environment where we verify and test it. For example, this could be a compiler, a unit test, a benchmark harness, it could be another LLM that judges the the code and the environment checks does this code actually work? Is it fast? Does it follow the format? Right? So we we set the criteria sometimes called rubrics

and the environment will judge based on that criteria and give back a reward. This is a a score. So the first implementation you see on the slide uses torch mapm. So it's correct and well formatted. So it gets plus 10. The second uses numpy which isn't ideal because it's not fast. So it gets a minus five. And the third one has a bug in it. So um it gets minus 100. We really want to penalize the the model that and tell it never to learn that. And it's this score that is at the heart

of RL. So we need a a valid signal telling us what's good and what's bad. So it doesn't really matter if you come up with the ultimate use case. you know that kind of codeex for healthare or or something that's nobody nobody's ever thought of. If you can't frame that in the context of a reward signal and deliver that reward signal back to the model, then it it's redundant. So it's really a balancing act between this high value domain and the ability to deliver that reward signal.

So one approach is in context learning, right? We can give the language model examples. We can take the scored outputs and inject them back into the prompt. So, so we're telling the model how to to perform. We're turning the task into a few shot one. This works, but there's a problem, a limitation from this. The process will create an ever growing prompt. You accumulate thousands of examples or many examples and the context fills up. So even models with million token context like we're start

starting to see now will degrade in performance. We know that models like Gemini the the latest Gemini model will degrade 20% on benchmarks that measure this like needle and the haststack. So this isn't something that we can rely on and ultimately it impacts the the you the usability of that model because you're relying on context to maintain performance and and fundamentally yeah we're just burning tokens and we don't want to do this. So we need a more efficient way and that's where RL comes in. So instead

of putting all of these examples into context, we assign the reward to every token in the output and the good implementation is scored with with say a plus 10 and every token in that generation gets labeled. So we're going to use these labels to update the model's weights through back propagation in relation in relation to these tokens. And for the bad implementation, we're going to do the opposite. We're going to give every token a minus 100 or whatever the score of the environment gives.

This is um this is very crude, but but it it generally works. And the the think tokens in this example were probably okay, for example. That wasn't necessarily a problem. Uh so we probably want to come back and fix that up, but I'll get to that later. Okay. So we can reframe RL in a slightly different way. So we can say that patience is all you need. We if we say that RL is just updating the weights for back propagation and using reward signals instead of supervised labels. We don't need a long

context and we can be more context efficient with in context learning. But the catch, the downside is that this takes longer than supervised fine-tuning. At the start, the model will produce garbage and get zero reward. So you have to wait and eventually it finds something good and the training takes off. If we give you an an example in a kind of geometric uh setting, right? Like just to kind of um help your intuition to grow. At the start of training, the model's output distribution is roughly

uniform. Every token has an equal likelihood. When the model produces these bad outputs, RL will penalize them. Penalize them pushing their probability down. As those bad outputs get penalized, their probability drops and the remaining probability mass redistributes across the landscape. So RL is systematically ruling out bad regions in that in that output space as tokens. >> Hey Ben, sorry if I may interject here just for one minute if that's okay. So you did mention about uh um how it's

different from supervised fine-tuning, right? Uh uh could we maybe take a minute here to sort of talk about what sort of scenarios would be suitable for supervised finetuning and where reinforcement learning based finetuning would shine out. If if you can maybe take a minute to point that out. >> Okay. Um yeah, that's a bit of a side note. I'm going to come back to supervised fine-tuning and where it fits in in a couple of slides. So, >> I'll leave it till then because we're

kind of in the middle of uh honing our intuition around how RL works. So, let's let's kind of keep focused for a second and then we'll we'll go to RL. >> Yeah, >> no worries. So, okay, we're we've got this um uniform this uniform um distribution, right? And we've adapted the uniform distribution by penalizing certain tokens. And once and the model is going to maintain this now um it's going to now learn about these tokens that it should penalize. At a certain point the model stumbles

upon a good answer and reinforcement learning will dramatically amplify this. the probability spikes up and it will um reshape the entire probability distribution. So this means that RL isn't isn't dumb in simple terms, it's just not very efficient. And so because of this crude per token uh reward assignment. So, so this is is a fundamental aspect of RL that we have to be aware of and we have to build environments that give rich reward signals back to the language model. So that the probability of a good

answer has to be greater than zero and that learning landscape has to be as as rich as possible. If the model literally can never produce a correct output or it's very difficult to ever produce a correct output because say the task is too hard, the formatting is wrong or it's too out of distribution for that model. Yeah, it's too hard, something it's never seen before. RL won't work. So this is the one of thing one of the things that we're really uh focusing on. Is this a task that a model

can do? Is it formatted correctly? uh and how can I format this task differently? You know, could I give it a better prompt? Could I give the task more information, more tools? These kind of questions you'll ask yourself. Otherwise, you'll just waste compute. The model might never learn. And common failure modes are wrong formatting, no pre-training warm up, uh and it's just too far from from something the model is able to do. So, that is where the earlier stages of the pipeline come in.

And so this is this responds to Bulk's question about supervised fine tuning. So in practice, we need all of the stages of the pipeline. Pre-training gives a language model a general understanding of of language, right? It's next token prediction. It allows it to pick up this general knowledge about how things work, how things relate to each other. And it's in supervised fine-tuning that the task is instruction tuned. So it will perform a certain task. It will learn to chat. It will learn to use tools and these kinds

of things. It will learn to use tools correctly. So it will be given examples of good usages. And in general, we want to invest the majority of compute in in this pre-training phase. And we want to invest another chunk of compute in the supervised fine-tuning phase because that's where the model will learn the right format and to follow instructions. So if we're planning to use a language model for something like healthcare, we would want to make sure that there was relevant information about healthcare in

its pre-training data. We could continue its pre-training data if necessary, it pre-training phase if necessary. And we would want to know that during the SFT phase that there was relevant instructions and tasks to its usage and we could go quite close to the line quite close to the task with just SFT. Let's say we had a specific task to use some healthcare tools. We could give examples to the model of how to use those tools as traces during SFT and the RL could squeeze out the last gains on

specific capabilities. Let's say another five to 10% on a benchmark for example sake. So um let's imagine that this is the kind of journey that the models go on. They start with pre-training and they move all the way through SFT and um and and into RL. And so it might look something like this. And and this is you know how you could imagine a modern training pipeline visually. So we start with a pre-training, we move to SFT and then we move to uh RL. Now, you could ask, could you just skip

SFT and go straight to RL from pre-training? Theoretically, yes. And there is a paper on this called Deep Seek Zero. But in practice, this is a very wasteful strategy. And the whole point of the pipeline is efficiency. So each stage should bootstrap the next stage. So we don't want to skip a stage like this and move straight to RL even though it's technically possible. Another way to make RL more efficient is process supervision. So instead of giving every single token the same reward as I showed you at the beginning,

you can assign different scores to different tokens. So in this example, you see that I've given the think tokens a zero because they're not necessarily wrong. The model is thinking between the it's generating reasoning tokens within the correct block. So you could say, okay, it's correct there. the real errors are are down in the Python function. So we'll use that for reward and and this gives a uh a much more informative uh learning landscape to the model. So we can take this further and we can

envelop develop these kind of rewards into our into our environment as well and give these um this more detailed process supervision. Another example of how we could do this is to use LLMs as judge. So we could say um is this line correct? Is this relevant to the original question? Um is this a concise answer to the question? These kinds of things that we can't necessarily implement uh as a verifiable function in code. So um then a major topic then is reward hacking. So the model might learn to gain the

verifier. It could delete the timer in a benchmark for example. I've seen that one myself quite often when you're getting a model to generate a say a kernel or an optimized piece of code. it will sometimes try to game the timing system within the code and find shortcuts to get around this. Um yeah, there's is this example about the the cobra effect where uh in India in colonial times um they offered rewards for for collecting dead cobras and so people bred cobras and then brought the dead cobras in.

This is kind of an archetypal example of um of uh of reward hacking. Maybe the original reward hack, but probably not. Um and when the the bounty was cancelled, the people released all all the cobras and and it was significantly worse before. The model will give it you exactly what you ask for is the is the simple takeaway, which uh may not be what you wanted. In the worst case, the model should try to corrupt could try to corrupt the testing environment itself. So it could overwrite files. It could modify its

test harness. It could inject malicious code. And so this might not in fact uh impact the learning landscape, but it could impact safety or security. And so this is why we need sandboxing for RL training. So in practice, you could uh you catch reward hacking by sampling outputs every 10 to 100 steps for manual inspection and use multiple independent reward functions monitoring output for suspicious rewards and having an untrained reference model as a judge rather than model judging itself. So the

the real takeaway from this is is to look at the trajectories, look at the interactions between the model and environment and just give it a kind of smell test, right? Just say, does this make sense what it's doing here? Um and and that will mostly help you to know whether it yeah reward hacking is taking place, but also whether your environment makes sense and and whether the game or the task that it's fulfilling is a logical one. whether it's one that the um the model is actually able to to achieve.

Okay. So, let's kind of level up a notch there and go to um how how this relates to the field. And so, we'll start to look at some more uh RL projects. Before we go into the environment itself, let's talk about how LLMs are typically post- trained. So most models today are post-trained in multiple stages as I mentioned SFT and RAR. If you want to go deeper into this, I'd recommend the Almo 3 tech report because it's one of the most open-source projects recently. It's from uh just

last year and they go really deep into this. So if you want much more information about the full life cycle of of an LLM, go and check out the MO3 tech report. What we also need to consider though is that when a a model or an agent is deployed in the real world, things get very messy. So, LLMs that are used in all sorts of applications, right? From generating codes to kind of booking flights and all these kind of examples. In typical in typical RL training, the problem distribution is predetermined by

a specific data set and remains static, preventing adaption to the policies models evolving capabilities. And so models trained with this static single turn data sets tend to be weak at correcting their own mistakes because the their reward was derived from single turn rollouts. In real life, models need to adapt to changing environments. Let's say they book a flight, but the flight is no longer available. They merge a branch, but there was another commit. These kinds of things. And so, one of the real

signals of a high quality environment is how it maps a particularly longunning task. So, that's something that I would strongly recommend in your project and will be rewarded uh within the hackathon. long running tasks with multiple trajectories, multiple routes through those environments. As I've said, environments can be anything. They can be web browsers, MCP servers, compilers, game engines. And a fascinating fascinating direction from the Chinese lab Kim K2 was to create LLM simulated APIs where a

language model pretends to be an external service and the agent learns to interact with these simulated tools. So you can think of environments as superersets of tools. If you teach models to use environments, they work better with real tools at inference time. So the idea here is that uh they're highly generalized environments that are generated interactively. So environments also solve the curricular problem. I mentioned earlier with uh static data set the problem difficulty is fixed at the beginning.

With environment you can dynamically generate uh problems matched to the model's current capability. So you can start off with easy problems and build up in complexity to harder problems. We could take a look at the RLVE paper that showed that training with over 400 verifiable environments significantly outperforms static RLVR which is uh reinforcement learning from verifiable rewards because the model always trains at the uh the correct difficulty. That's a really interesting takeaway from that aspect. So as I

mentioned earlier, if the model is never achieved this reward, it never gets this signal, then that we're wasting compute. So what we can do there is to introduce easier tasks at the beginning and build up the complexity of those tasks so that the model is constantly getting this reward. For the example of um booking flights, let's say that in the first few turns of the environment, they're quite simple flight situations. Okay, the flight is available. There's definitely one to choose from. Uh and it's given a

reward for booking the correct flight. Later in the training in the training phase, it might be given situations where there is no flight available and it needs to go back to the user to request more information or there are multiple flights available or there are there's no direct flight available and so it has to um use a transfer. These kind of complexities you can introduce as the process goes on. And and one important thing to mention is that Frontier Labs are already scaling this aggressively. If we look at

the Deep Seek uh V3.2 paper, it used nearly 2,000 distinct environments and 85,000 prompts. So this is a huge part of training frontier models. Now is the is the takeaway. We can also look at at the miniax paper which used over a 100,000 environments extracted from real GitHub repositories and each repo became an environment where the agent gets a task like fix this bug and it navigates the codeace fixes issues and runs a test and so the the testing suite in that was used as a as a reward signal.

Okay, that that's really cool. Those are some examples that you can take go away and check out if you want to deep dive on this. But if you actually look around GitHub, you look around the ecosystem, what you'll find are a lot of RL environments in GitHub repos, um, very difficult to to find and and not compatible with each other. And it's kind of comparable to the early days of of language models actually um and before they were on the hugging face hub and and when they were in kind of Google

Drive and they weren't necessarily compatible with the training frameworks and they were difficult to use. So because of this they're very difficult to scale. They might be quite hard to even set up as a single training run. But to get up to the thousands that frontier labs are using is almost impossible or probably impossible. And that problem is what motivates open. We want RL environments that are for everyone and that are compatible and have a single interface so that we can easily scale them.

So open M takes a simple strategy where an agent interacts with an environment. The agent makes actions in the environment. It gets back observations from the environment and a reward. It uses a single API that's consistent across all environments. And it allows environments to be plugged into a training run on mass and that training framework be able to handle those due to their interoperability. So here's a concrete example. Not the kind that you could win this hackathon with because in itself it is quite

simple and we're really looking for real world ones, but it's a nice uh educational example. So we train a model to play the game Wordle using GPO. If you check out the course, you'll also find the example for this if you want to follow along. the course being the one that's linked with the the onboarding material. And in the Word or environment, you define a roll out function of how to play the game and how to generate guesses and feed them to the game and extract rewards. You'll notice here that the game gives

partial rewards. So, in the game of Wordle, uh, if you get a yellow letter, it means that that letter is correct, but it's not in the the correct place. If you get a green letter, it means that it's the correct letter and it's in the correct place. So, uh, in this environment, when the model gets a yellow letter, it was given a 0.2. And when it gets a correct letter, it's given a 0.4. four I think off the top of my head. And so what this means is that as those very small models that we use in this train

example of a seven bit.7 billion parameters, as they struggle to get correct guesses, they get a certain reward for guessing the correct letter. When we trained this, what we found was then the language model would start to predict the correct letter again and again and again, which wouldn't really make sense because if a human got the correct letter, they would then think, okay, I need to now move it to the correct place. So they would move it and and it would just keep guessing say one word, say crane, crane, crane.

So we penalized the model from for repeating the same guess and that gave it more signal that it could use and then it would stop repeating and it would start to make new guesses and once it made new guesses it would move the correct letter around and and it had a richer landscape. And we can see in this example that you have a valid reward signal. This is as I said a basic example. It's something worth trying out. It's not the kind of environment you could submit to this hackathon but uh it's definitely a great educational

one. As I mentioned, but maybe it's worth iterating. Open M is framework agnostic. So the example I showed works in TRL. Uh there's also onslaught examples in the main GitHub repo, but it works in other libraries like agent reinforcement trainer, lightning AI's library. It works in Umei and others. And the environment spec is the contract. So you don't need to set up integrations with any library, right? you can just define the environment and plug it into that uh training framework.

So let's take a look at how it works. Right, I'm in a second going to go over to hugging face and show you what it looks like there. But for now, I'm just going to show you it on these slides. So um every environment is a Python package. I'll show you how we make that in a second. But what that means is that we can install the Python package with pip from hugging face. We can say pip install the Python package and we'll get the client code and as you can see here we can say mv equals my and then we can

pull it from the hub as a docker image or we can interact with it with the hub running there. So that's why the space is quite convenient. We get both the um we get both the server, the container and the client code all in one place. And this is what it looks like on hugging face. You might have seen this already. They're small apps and they're based as spaces. So they're deployed with a UI and an API. They're also a git repository, which means they're version controlled and you can install them. As

I said, they have authentication, so you can have a private end. They've also got a Docker registry, so you can pull it as as you saw. And all of this is wrapped into the OpenM CLI. So, in order to create an M, you can just do OpenM in it, give it a name, and then OpenM push, and it's deployed to the hub. Okay, let's go one level down and talk about how we implement the amp. So I'm going to walk through what the skeleton contains. So there are five files, five steps um which you define in models.py. So that

would be the action, the observation and the state which are as pyantic objects. The second is to implement your environment with a reset step and and state. The fourth is to create the the HTTP client and then the to wrap that in in fast API and the final is to dockerize it. In reality, all of this happens with open-end in it. So you don't need to go into these details, but these are the five components that bring the M together. When you do open MV in it, you'll get all of this out of the box.

And so here's how the full pattern would work with something like connect 4. Your action data class is a column in the game connect for your observation is the board and the legal actions and done. So let's say the rules and the state tracks the board and the next player and any player moves. These pedantic objects extend the base open M types. So you're able to use those from the core library and they get packaged into docker into the docker file. The server will expose a reset step

state and health endpoints which are all necessary uh for it to work. And here's an example another example with a GPU kernel sandbox. So on reset, it detects the the GPU model that we're working. It runs a baseline, records the baseline, and then on step it receives some kernel code, some code to be optimized, compiles it, and then runs the benchmark, measures its performance, and returns back the reward. So the code for this kernel example follows the same pattern as as connect 4. You have a kernel action kernel code

uh etc. And it works through that. So what you'll see in your environment is this the same API naming structure with with action observation etc but wrapped around your task. I'll show you that in a second. So that's basically the end of of the slide. The one thing I'd say is to check out the GitHub repo, especially if you want to take it to the next level. We have other features in there like an MCP interface and uh rubrics for supporting LLM as a judge. So, I'd go and try those

out. Now, I'm going to show you uh in the IDE what this looks like, and I'm kind of open to any questions um as I transition across. Cool. So, what you'll see here is a just an empty directory, basically an empty folder. And um I've got a Python environment. So, I'm going to install OpenM. The OpenM library is called OpenM Core. uh and it has a group for the CLI wrap that quotations and we can do this. So it was already installed in my environment. And so what I can do here, this is a bit of a of a

trick. If you want to um if you want to use an agent for this, you can install skills with the CLI. So you can do open skills add and say I'll do codeex as a parameter and then it's created a sim link. I'll come back to that in a second when I show you how to use an agent. But if I want to create an environment uh my of my own I can do open m in it uh kernel m. So I'm giving the m a name. So I'm going to uh create a kernel optimization environment. So I say open init kernel enter.

And what you'll see here is that the CLI replies it created 11 files generated a UV lock and it created the environment. I can take a look in that directory and you'll see that I get a server with an application file in it. Manages the imports for me. sets up a just a function that runs uh the Uicorn server inside this directory inside this file sorry it will then create my environment and it's used a template and dropped my name into it right so all the way through you'll see all of this preset up and in fact you

could use this environment it it's very simple it just rewards based on the length of the message so you you can't really train anything meaningful but it is a working environment out of the box and so that makes building uh much easier. You'll also notice that there's client code there and the models to define the action. So in this case the message and the observation back is the echoed message and the message length. You'll notice that there's a openm.yaml YAML which is where we configure how it

works. Uh it has a a PI project toml. So the environment itself is a managed Python project. And you'll see that all of the dependencies are installed there, but you'll need to add your own dependencies depending on the project that you use. If you use Python dependencies, that's fine. You can just put them all in here. If you have any non-Python dependencies, you'll need to use the Docker file. Uh, and that you can do here in the in the Docker file in the server. Now, as I mentioned, you'll also need to

um you'll also be able to use a coding agent to do this. So, let's take a look at that. I'm going to now delete my M I just created from the template and I'm going to open codeex and I'm just going to reference the skills to double check that it's there. Yeah. So there it is. I've got this openm skill. So I'm just going to tell it to use it to make sure it does. And I'm going to say create an environment for benchmarking kernels. Let's give it some more information. Let's say PyTorch kernels.

Make sure it knows it's an RL environment. Uh cool. It should know that it's open end. Let's take a risk and not tell it to use open end. Okay. And now that's going to get to work. And while it's doing that, I'm going to show you how the uh skills are packaged. So what you'll see here is that it uses a skill.m MD and it it's all referenced in here. And so it has an instruction of how to use the same CLI that that we used there. Great. So it's quite simple. We can also

go and check um the repository online to show you what it looks like. So if I now show you an example of an environment on the hub and that will finish up. So if you take a look on hugging face and you go to spaces, you'll notice that there's a filter here called agent environment. If you go there, you'll get a series of around 900 almost a thousand RL environments that you can be inspired by. They use different libraries. You can also filter it so it's just open M's. Uh, and so that's what you'll see here.

And they're these are all implemented in OpenMs. Or you can go to the openm.org here and you'll get a list of the MS that that we're supporting. So let's take a look in there at some of the you see there's a collection here got a number of environments there and what you'll see is that this is an environment on hugging face which a space on hugging face and when it's deployed it comes with its own uh standard template that tells people how to use it that's all packaged in and

a basic interface of how to use it manually to test that it works. You can also add a custom interface to the Gradio install and and that's listed within the documentation. So it it's quite simple. It's just an extension to the default UI that is specific to that game. And that just helps people to use it. Like if there's quite a complex environment that you need to debug like Wordle or or anything else, you can use Gradio to create these custom UI components. I'd recommend doing that if

if you want to build up high quality MP. And so that really concludes it. Let's see how Codeex got on. So it wasn't able to finish in time. It's still working. But what you'll see here is that codeex is conforming to the correct in fact it is great. So it's already here and what you'll see is that codeex is conforming to the standard because we've given it the skill that it needs to work from. So uh sorry I'm not sharing the correct screen. So it codeex is still working here using

the skill to generate the environment that I showed you just now. And if we take a look in that environment we'll see that it's no longer just a skeleton. It's actually a working PyTorch environ working kernel bench environment that measures the performance of a kernel. Now, it's generated by an agent. So, it could very well be errors in here, but compared to examples where agents don't use skills, we found this to be a huge leg up. So if you're using any agent, cursor, codeex, claude, open code, I

would definitely suggest that you install the built-in school skills from the library and they'll tell the agent how to build an openend environment and mainly tell it to use the CLI to start off from and that gives it the API contract to begin with. Okay, so that concludes uh my contribution. Uh I've got some time for any questions that jump in. uh if pulkit wants to share them but I'll mainly hand over to PKit now just >> [snorts] >> Okay. All right. So firstly, thank you

so much uh Ben for that uh walk through for that introduction. I think that was super super helpful uh for the hackathon. So what I'm going to do now firstly we'll just take a look if there are any questions for Ben that we can share and uh if not I'll now give a walk through of a sample submission that you can do uh after going to the hackathon page. So that's one of the things that we want you to walk uh walk away with after this particular session. So yeah, let's just quickly see if

there's anything. So one question here that I think is quite important is why is the inference script mandatory? That's a very important question. The and the reason that it's mandatory is that that's how we will evaluate the environment. So I I mentioned that at the beginning. If a model is never able to get a reward on an environment, then that environment is basically useless to that to that training process, right? Like we need to have a balance between something that is valuable and something that is is

feasible. So we'll use the inference script to to measure how valuable uh that environment is. Can you share the skills link please? There is no link. So if you use the OpenM CLI, you'll just use the command openm skills add and it will add those skills to the directory you're working in. Check out the openm documentation for that. There's a doc there's a page on how to add skills or just do openend help uh in the CLI and and you'll see a command for doing that. Okay. So uh one question is does the GUI

development support 3D environments? It really supports anything because it uses Gradio. So Graddio has components for kind of anything you could really imagine. There's a HTML component which is obviously very generic and you can use JavaScript in there. There's a a video component and and other components. So, Graddio is extremely extensible and you can add any of those modules into the Gradio application. Okay, I'm really focusing on questions that that are more about OpenM, but I

can see that there are some questions about here about the hackathon itself. Okay, I think that's most of So, we need to integrate an OpenAI API into our environment. No, that would not be into the environment. You would use the OpenAI API within the inference script and the environment would not have any mention of that at all unless you were using a judge within the environment for example. Okay, one question is so we can't use any other agent other than codeex. Uh, no that's not true. You can use any

agent you want to generate environments. Uh, but you do need to use the open AI API in the inference script because that's a standard and we can easily swap that with different inference providers. Cool. I think that's that's a nice set of questions there. Um, I'll be around in the Discord as well if there are any there. Definitely check out the Discord and I'd go to the GitHub repo and check out everything there and give it a star. Um, cool. That's everything from me. Thank you so much for that, Ben. Again,

um I think everybody really appreciated having you on and giving a walk through of uh the entire ecosystem as well as how hugging space helps in doing the deployment uh of the model and uh yeah thanks for taking up that question on inference that was super important. I'm going to sort of highlight the same thing now but uh yeah so what I'm going to do now uh just to keep things very simple let me firstly take over the screen if that's okay Ben and uh I'll I'll take you all back to the page so

this is what we are going to do now uh in the next part of today's uh live first share my screen. All right. I think my screen should be visible. If I can quickly check with the team um team, is my screen visible on the live? Perfect. All right. Okay. So, so uh once you register for the hackathon, this is the page you land upon, right? And over here, uh it's it's a specification for everything that needs to be done and how the task essentially needs to be completed. What we want to make sure is your uh first

round first step for round one is well in place, right? That's what we want to ensure and uh how we can help you out with that. So Ben has given you a walkthrough of the entire thing how the environment needs to be built and how it needs to be deployed. What I'm going to do I'm going to take a very very simple echo environment right just to keep things very simple because that's something that goes with the inference script that we have provided. So one of the challenges that we believe is uh how

do you essentially use the inference script. So to ensure that you're able to use it on the dashboard itself, you should be able to see the inference script over here, right? So that's what essentially you can copy and put it in your project and uh that's what you will be deploying and that's how we will be assessing your environment. That's the whole idea. Okay, I'll give you a walkthrough of this info script as well. But let's start let's let's sort of give

you a quick recap of everything that Ben uh sort of said, right? Just a quick very quick recap not in terms of the theory but in terms of how practically you uh accomplish this task right. So starting from the setting up of the CLI you have to do that and uh then moving towards uh creating a very very simple environments an environment that gets set up straight away after you do your open enit command right so make sure all of those things are well set up you also need uh docker running in your local

right that's how you will be able to proceed with all the different steps that need to be taken to make sure this runs smoothly Okay, another thing that you need to do is because the deployments are happening on hugging face hub, right? So, make sure you have a token that you have created for yourself, right? If I may check with people in the live very quickly, have you guys generated your hugging face token? Are you guys all set with the hugging face token? Okay, in case not in case not. So once

you log in, you have to just just giving a quick walk through, right? So just go to the access tokens and over here you can go ahead and create a new token and that should be that should that should do it, right? And just make sure you are keeping it safe somewhere because once you lose it then you'll have to just create it again, right? It'll be available for you to see only once. Okay. So once this is done, right, once this is done, we come back and uh let let's quickly proceed through all the

different steps that need to be performed, right? So let me share the different screen now. All right. So you know some uh sanity checks just once you're on the uh terminal just make sure that things are working fine your open v open envis setup those things are something that we need to make sure of right so uh wherever you plan to create your uh project just what what I would highly recommend is uh build a virtual environment and try to do things within the virtual environment right once you're done with the installation of

open in right so you have your pip install right pip install of open n core right once that gets done then your library would be set you can quickly check if everything's working fine by entering your python right so just try importing the library okay so this gives you a signal that everything's set and once Once this is done, your CLA should work absolutely fine. So to quickly do an open in it, you just need to create some environment where you will be setting up your project. So let's say

RL demo. Any name is fine, right? I'm just doing this to um make you guys sort of be familiar with this. Zooming in a little bit more. And once you do this, okay, so uh this is a signal that everything's worked fine. And while that happens, you should see over here all it also highlights all the steps that need to be performed post this to ensure everything is working fine. Right? So now uh ju just just to give you a quick scan of what happens once you do this. What it does is it ensures all the files

are set the way they should be. Okay. Now one of the ways to show you if everything's working fine. By the way we have shared this link on the live. You guys must have received it. Okay. Sharing a different screen. Just give me a second. Okay. So over here if you sort of go into by open end like just just into the intro part it shows you how the entire thing is organized and what are the some of the things we need to be aware of. So firstly uh you should be able to see over here that this is how so as I

created my environment now the great thing is that all of these things get automatically created right everything gets created automatically one thing you need to do is make sure you move this docker file outside right it should not be inside server just move it outside within the main folder uh that's one of the things that would be important to do okay and the other thing is this models py is the file within which all your uh functions are going to reside. Right? So basically everything that's connected to

your reinforcement learning loop. Okay? All of those definitions are going to reside within models and that's where you can go ahead start doing the updates and start updating your environment. So to give you a walk through now I'll take you to uh my VS code and over there I'll give you the walkthrough. This is essentially what you'll observe once a project gets created. So let me share that again. [clears throat] All right, you should be able to see my environment now. So yeah, so this is uh

a project that I've created and once this is done, you should see all the files well within it. So as you can see from here from server I have moved the docker file outside. That's very important thing to do. Uh another thing that you can do if you if you remember from uh Ben's presentation there was a clean web interface that we were able to see to ensure that you are also able to see that web interface. One of the things that I recommend you guys to do is to put in this line right which is n enable

web interface equal to true. So this way when you call the API right you will be able to see this interface and uh that that that just makes things easy you know that everything's running fine and you can even do some small experiments over there. Right now again highlighting very quickly if you go to the page if you go to the dashboard it's very clearly stating that you have to create real environments not toy environments. The only reason why I'm showing you right now everything with a toy

environment with the echo environment simply is so that you are clear with the relation between the inference script that you need as well as the environment that you create because you'll have to update the inference script accordingly. Okay. But right now I just want to make sure that you understand the connection and it's all running smoothly. Okay, that's all we are trying to check right now. Okay, great. So now uh once we have the uh once we have added this line to the docker file and uh

circling back just to give you a walkthrough of that models. py as I was mentioning right. So uh this is what your init command does. It automatically initializes everything, right? It makes things super easy. So you can see over here that all the different things that you do in your reinforcement learning loop or the iteration they're present over here right. So this is the action and this is the observation. So both these things are present and it has created it for you. So this is what you

need to update while you're creating your environment. This is where you go and make the changes. This is a super simple environment called as echo. Right now I'm going to keep it as it is because I want to make sure you're able to understand how everything's working. But yes uh at the risk of being too repetitive I just want to call out again no toy environments for experiment it's fine but you have to go ahead and work with real environments. Okay great now um and also uh towards the end I will

share some uh links with all of you of some real environment so that you know it it gets easier for you to work with this. All right. Now once we are uh done over here, once you have made the necessary changes in the model file and uh you can sort of now start running everything right in the terminal itself. So pull up a terminal and start executing all the commands that were mentioned when you ran your init open in it. Right? So let me share a broader screen so that you can see multiple things at the same

time. Just give me a second. Okay. So if I'll take you back to the terminal, you can see over here we have all these commands, right, for uh creating a Docker. Now just make sure your Docker is running and then we start with the Docker build, right? So now you you have to make sure that everything aligns with the name of the environment that you have created. That's super important. So I'm going to do the same. I'm going to make sure it aligns with my uh my environment. And also ensure that

you're using uh you you also have UV installed. That just makes everything smooth and simple. Uh if you remember from Ben's presentation as well, he used uh the UV to run everything, right? Even the inference files. So coming back. Okay. Um so firstly again pip 3 install u for me it's already done but you make sure that this step is performed. Once this is done we are now ready for the docker build. So you go to docker build. Okay. Oh one second. I think my screen share stopped. Yeah just a second.

Right. You should be able to see my screen now. Yeah. So, we are basically running the docker build command. Okay. So, we'll say docker build. Uh this is the same command that gets generated for you. Uh make sure you're naming it as per your environment name. That's very very important. Okay. So this should create the docker. Okay. All right. So once this step is done, once the docker is created, now we'll go ahead and uh sort of create the web server, right? So we'll launch the

server. So we'll go ahead and run it. So, do we run right? Make sure you're writing the name of your environment correctly. So, I'll just copy from here to make sure there's no mistake. Yeah. And let's hit enter. Yeah. Okay. So, this is a common error that you might get if you're running things multiple times, right? So, this is basically happening because uh for me the docker is already running. So, I just have to make sure that I shut that one down. Let me quickly do that.

Just give me one second. Okay, let's try it again. All right, works fine this time. Let's go ahead and uh launch it. Okay, let me share the other screen now in the browser. Okay. So if everything's running fine, you should be able to see this, right? And uh sort of to again make sure that you're clear with this that there are these three essential steps that you need to be clear with the step, the reset, and the get state. Right? These are the three things that we need to be familiar with. And for each of these you

can essentially this this interface which you get at uh /web right so essentially when you launch it you you'll be on this particular screen but then you can go to the web and just open it and once you're here you can just uh plug in a simple message because the purpose here is just to check if it's echoing back right so if I just say hello click on step and you should be able to get some output over here and you can see depending upon how everything's been defined we getting some reward, right? So, this is

how your entire thing should work. Okay, great. Coming back to Visual Studio and sort of now tying it with the inference script because that's very important. So, let's let's quickly do that. Okay. So now taking you to the inference script just to sort of make sure you're clear with it. This is an important step that we all need to align with because even if everything's working fine, you might face trouble with your submission unless the inference script is present. So quickly calling out few important

steps. Make sure the docker file is in the outermost folder, right? Don't let it be within server. Okay, that's point number one. Point number two, you also have to create this inference script. Now this inference script is coming from the dashboard. Okay, I have simply copy pasted it. Exact script is going to work over here simply because it's associated with the echo environment, echo demo environment that we are creating. Okay, but if you're making changes, you'll of course have to make changes to the

inference script as well. Now here what I recommend is because this is this script essentially is the format in which the submission needs to happen for whatever real world environments you create. You feel free to use AI to help it help you to you know configure this inference script and make changes to it accordingly. Right? So just provide your entire environment. Whatever changes you do to the model file, whatever changes you do to the environment file and accordingly you ask AI to update your

inference script as well. But to ensure that you have everything in place, make sure you're reading through all these steps over here. Right? Make sure you're reading through all these steps. Okay. Uh one more thing by the way, uh your open validate command should also work if everything is in the right place, right? So just like we have the init command, we also have the validate command. So once you have everything in place, run validate to make sure it's all running fine. Okay. And uh yeah, so

now over here there are a couple of things that you need to make sure are uh in place, right? If I sort of take you down, you will see over here we are doing some initial setup. So make sure you have everything aligned over here in terms of your docker image name, right? So all those things need to be provided over here. The image name, the API key is essentially the hugging face token that I sort of showed you just some time back. Okay, this is again super important. Um U u over here you can see

I've created this n file where I'm sort of putting everything all my environment variables are going into this one. Okay. Now yeah once this is done you specify the model as well. Uh also make sure uh of a few more things right you you do provide the name of the model and updated over here if you're using a different model right also another thing if you notice on the dashboard it's mentioned over there that it's the open AI function that you really need to be using for this

particular submission. So make sure you are not messing with that part as well. make sure it's in the right place and uh once once you're taking care of all those things then it should all run fine right that's the idea like you can also test your inference script so that's what I'm going to do next but uh after giving you a quick walk through of this entire script like what we are doing and how you will go ahead and uh make the necessary changes right so this inference script is sort of running that

inference loop for you right all the different steps that you're supposed to perform it's doing it for you we are creating a prompt the library that we have used over here the textrap library is just simplifying for us how to create this system prompt that we are uh passing that that's the entire purpose of this okay and uh just make sure of one more thing right just make sure you have the um you you're executing the script towards the end, right? This is again a place where uh if if things are not in the right

place, you might face some issues, right? Okay. So, let's go ahead and uh run the script, run the inference file just to make sure everything's running fine, right? That's the final thing we'll do. I'll just open a new terminal for that. Okay, just allow me a second. So, we run Sorry, just a second. Okay, let's quickly resolve this. Give me a second. Okay, so there are a bunch of uh steps that have to be done before we uh go ahead with this, right? one, we quickly set up a virtual environment

within which we execute all of these steps, right? So, let's let's quickly do that. It's it's like a full-on uh initialization of everything one by one. >> Okay. I think the screen share stopped. Let me do that again. Okay. I hope the screen is visible now. Are you all able to see the screen now? Yeah. Okay. All right. All right. So, let's just make sure we running all the steps one by one and uh it's a full initialization of everything. So, Okay, let's activate our virtual

environment that we have just built. Yeah. So once you're inside the virtual environment, we now make sure that some of the things that we need to be able to execute this are installed. So we'll do a quick installation of the UV. Okay. Let me make sure everything's clearly visible. Yeah. Okay. All right. So once the installation of UV is done, we go ahead and do UV sync. Next we make sure our environment variables are in place. And finally, we go ahead and uh launch the inference file. Let's take a

look at that. All right. Okay, you can see over here uh the steps have started, right? This is a confirmation that everything's running fine. All right. So once you are at this point that's that's an indication that your inference script is in sync with everything. It's all running fine and uh now you're good to go with your submission. Right? So that's that's the final thing. Uh when Ben covered his presentation I hope you do remember that all of this needs to be pushed to

hugging face right hugging face spaces. That's where it all goes. And then finally you share the link of that. So just just to give you a quick walkthrough of the final submission steps once right let me take you to that page right let let me share a different screen take you back to the browser Okay. Yeah. All right. So, uh this is the script that we use. Now, just to give you a final walkthrough of all the steps, right? So uh for this module 4 this is the walkthrough that we gave by the way for each of these steps that you

see what what Ben and I have done now sort of let's say if there are some some things that you are unable to connect you feel you you need a recap of things within each of these modules we have videos embedded which are sort of a walkthrough of these steps right so all the steps that we have taken and uh the focus point of step one of the submission is building up our own environment. So what I just gave you a walkthrough of you'll find that over here and you will find it with respect to the verdle environment. Now I gave

you a very simple walkthrough right? I gave you a walk through of your echo environment. We we kept it super simple but to show you the kind of complexity that you can look into while building these up. Uh I I'll take you back to the course. We have all shared this link of this open end course that the open end team has created. I'm I'm talking about this particular uh course and over here if you sort of go to module four right if I give you show you the notebook in the module 4 you'll see it is with

respect to the word example that Ben talked about so that just shows you how you can create something more complicated right so keep taking tiny steps you take the first step make sure echo runs smooth but again please remember that's just a toy example that submission won't count that's not a good enough submission then you can look can do some examples that are way more powerful. Let me do one thing. Let me also share with you um a couple of links that will that will sort of show you

what are the things that you can do with this. So let me let me share a couple of links with you guys. Uh the kind of environments that people are building. Now again please keep in mind we do have plagiarism checks. So uh do not entertain the idea of you know reusing any of this work. you have to create your own environments along with the influence script. But just to get a sense of the environment itself, some more examples for you. Ben has already shown you some examples on the hugging face uh openend space itself. But I I'll

give you a few more. So team, if we could share it on on the live that would be great. Sending it out. Just give me a second. Okay. So you you'll be receiving the link soon uh on the live itself on the chat. Okay. Now um yeah as I said you can go over this particular uh module to see how the environments need to be built. You'll see over here how they are being how the different classes are being modified right especially for a for a task or for an environment that you are trying to create. So make sure

you uh go through this right now coming back to the hackathon page. So once you go through these courses they will help you with a quick recap of everything. Then okay as a as a final piece just remember this part. This is super important. Uh once you are done testing locally this is what we did right we did uv runs inference. py that gave us an indication that everything's running fine. And once you're ready to go you can you have to do this open push. Now if open is working fine if you were able to do in

it you should be able to do the push part as well. And doing this will help you to push your environment to the hugging face. Right? So all you need to do is uh put in your hugging face username and the name of the environment and that would do the job. After that if you go back to uh hugging face you will be able to see that it uh environment deployed. And then the final step is to be able to share the URL of that space over here. And that would be all right. I think that is all from my side and we

will take questions now. All right guys, just allow us a couple of minutes. We are accumulating the questions so that we can start addressing them and also you know just making sure if we have covered everything that we wanted to so that you guys walk away from here uh with some confidence with some idea of how to make these submissions. You still have the time right? Uh the deadline is you you still have a good week for the deadline. Okay. Okay. So uh once again highlighting the fact that this script is

what will enable the evaluation for us right without this script the evaluation won't work okay and this is exactly the script that I executed but you will have to of course make changes to it depending upon your tasks right you can see over here this is where we are configuring the rewards and uh this is what is validating how well is your uh RL environment working. So make sure you are updating this as per the environment that you build. Okay, cool. Just just make sure you are following through all the mandatory

steps and don't don't miss out anything. You can see over here the mandatory steps have been called out. So make sure they're all in place. Without this things might not work. Okay. Uh especially when it comes to your submission. uh while they while you might feel things are working fine in the local but without this the the inference at our end when we are validating your script and when we are validating your submission we won't be able to do it without these mandatory steps. So just make sure you're

following through all these mandatory steps. All right. Okay. Perfect. Um maybe let's take a look at the questions whatever are there and try to sort of uh answer. Okay, I think this one is an important question around uh the API itself, right? So, make sure you're using the hugging face uh token. Uh you don't have to pay for the openi token. Do not create the openi token. Again, I want to quickly highlight one thing. If I take you back to the script once. Okay, I think the script page is already shared.

So, if I show this part to you quickly, you can see over here, right? When we are mentioning about the API key, we are basically loading it from the HF token. So, that's the idea. Even if you do not have uh OpenAI token and we do understand that it's paid, so you don't have to worry about it. uh from hugging phase the token that you get you get some free credits and that should be good enough for you to uh experiment with it and make your uh validate everything and make your uh submission.

Right. Let's see what other questions do we have. Yeah. And the other thing would be over here uh for the API based URL use the router. The the idea is as follows that uh what this essentially does within hugging face all the different um all the different large language model providers are available right so that's why we are not expecting you to call these externally. OpenAI claude all these environments are available. So just use the router and use the hugging face token. It will automatically take

care of things on your behalf. You don't have to worry about anything. Okay. Cool. All right. We have I if I see the questions as I said we have already called out that you don't need to use the open AI key. You can use the hugging face token. That would be good enough. Yes, I think Brahan RK has called this out well that FF router is a auto select model feature and uh that would that would take care of calling the models so you don't have to worry about it. Okay. Okay. As far as the selection of model

goes, again feel free to use whatever you want. In this particular example, we used Quinn uh 72 billion instruct, but you are welcome to use something else as well. Okay. All right. To quickly recall, so I do see some questions around the uh around the perspective of overall wider perspective of the hackathon. So guys, focus on task one, which is to be able to create our own environment. See understand in reinforcement learning what you do is as good as the environment you have and the idea for

open en package is to be able to give you the opportunity to create environments as per your expectation right for the link that we have shared with you earlier you would be able to notice that there are some very complex environments that people have created and the idea is we want you to be able to create your own environments okay later on we'll come so once you're into the stage two then we will start talking about things around how do you essentially execute everything but right now the focus is build an environment

build an inference script for that environment right that's what the focus is for uh this particular round okay Yes, absolutely. Feel free to use AI to your advantage, right? Um, especially for updating the inference script, right? The inference script that we have is for the echo dummy project. But when you build your own environment, you would have to update it and uh feel free to use AI to make that happen. Okay, this is something that we have tried at our end and it does a fairly good job at

updating as per your environment file model file that you have created. It does a pretty good job at updating the inference script. Yeah. So, uh on on the topic end, right, somebody said the inference script should be associated with some topic, right? So that's a very legit question. Please understand the environment you're building is itself the uh topic. Okay. So let let me clarify with a simple example. Okay. Think in terms of let's say a number guessing game. Right. Now uh when we talk about a number guessing

game. So please hear me out for a couple of minutes. Okay. If we talk about a number guessing game, the idea is for your large language model to be able to play that game well. Right? And based on that we want to reward it or negatively uh give give it negative marking if it does not do the right step. And the idea is the game is as follows that you have to guess a number between let's say 1 and 100. And uh you have to ensure that it is able to guess the number within 10 steps. Right? Let's keep it simple. So

now you can actually build this environment. You can create this environment. Right? And once the environment is created then you can create an inference script to test how well the environment is working. So when we talk about test this part is not linked to getting a high accuracy. It is simply to check if it is working well as per the grading scheme that you have created. Right? You have to ensure that the grading scheme is for giving a marking between zero and one. Right? And you can you can create your rewards as

per uh the requirement of your task. So the idea is now just think about it right what would you want you would want this model to essentially be able to learn to do uh something that quickly comes up with the guess right so if it's just guessing randomly it's going to be very challenging to find the number but but if it is able to build a strategy it should be able to do it but we are not worried about those things at least in this particular step right in in the first stage in the first stage we are

simply concerned about you being able Think of a problem out of the box problem and then be able to create the environment for that problem. That's the focus point, right? Okay. So for all the questions pertaining to the ability to use other models, I just want to clarify one thing. Just make sure whatever you use, you use it through the HF router. That's super important, right? because that's the token that you're using and because your model ultimately gets it's all connected to hugging face. So just

ensure that whatever model you use is a model that's available within hugging face that that's the only constraint but otherwise within that environment you're free to use whatever model you want to choose. Yeah, sure. Let let me share the example that I just mentioned. Right. So for the number guessing game for example like I was talking about creating it on our own but that is present by the way uh as an environment already in hugging face spaces right somebody has already created it so you can very well uh use

that okay so for the grader the grader should return a value between zero and one. That's the whole point, right? You can see over here. Uh, okay. Team, uh, can you share back my screen? I think. Yeah. Okay. Perfect. Yeah. So, if you are able to see my screen, you should see this part, right? The grader should return a value between 0 and one. That's the whole point. Okay. So, your revolved system should be created accordingly. >> [snorts] >> Okay to to make sure people understand

the inference script again just quickly calling it out. This is where you need to make the changes right. Specify the model that you wish to use. These are the mandatory parts of the script. Okay. Please make sure you are handling these right. Um the base URL I suggest you to stick to the router that takes care of everything. For the model name, uh you ensure what model you wish to use, but it has to be from within hugging space. Just take care of that. Okay. Uh you can update your token while you're working

in the local and uh also the name of the image with which we build the docker with. Okay. So these are the things that you need to change. Okay. also getting some questions which are I mean not directly connected with the um walkthrough that I gave but there's a question on uh how many teams will be selected for round two do we do we sort of have that decided upon or is that something that we are that's something that we have not yet um revealed and we wish to keep any any thoughts from that team?

Okay. All right. So, uh basically the idea is that there's like so whoever does well on the task can expect to you know uh follow through on the round two. And uh the idea is just build environments what you are being graded for that's again available here on the page itself. Right? So if you quickly take a look the I would also request all of you to treat this particular dashboard page as your uh as as the holy grail for everything. Everything's mentioned over here. Right? Just just make sure you're you don't

miss out anything from here. The greater part is mentioned here. Okay. If I talk about let's say the judgment part itself right so you can see over here that uh we we have called out do not do not plagiarize and uh make sure that the grader has some diversity in it. It should not be returning the same value or the same score every time. So these are the kind of things that you need to be aware of, right? Okay. Uh there there's also a question on how many submissions team can can you help

me with this? Is it one or do we allow multiple submissions? Okay, we are we are open with multiple submissions guys. So just do as many submissions as you want. No problem. Okay. Sneha that's a good question. Again both matter and we have sort of created a We do have over here like a like an example that you can refer to on the dashboard page. Okay. And I'm specifically talking about your question here which is that what matters the idea or the everything works fine. It's a collection of both things

right. So you have to ensure that your inference should work fine. That is important and also how creative your idea is. Both both those things matter together. It let me quickly show it to you. Yeah, you can see over here right the criteria. So the utility of the idea. Okay. And then uh the quality of the grader that you have built the how well have you designed the environment and uh yeah that that's that's how the scoring is. Okay. Okay. So, you can make submissions only only up until 8th of April, guys. Please

also keep the date in mind. That's super important, right? Uh you can see over here that uh deadline is 8th April. All right? So, please don't forget that part. Okay. All right. And another thing um so if if you are making multiple submissions and multiple of those are correct the latest one is what we will be considering. Okay. So your most recent submission is going to be used for the evaluation. Right. No, Ki, I'm again specifying super super important. Your submissions have to be

real world. I took the gamified example simply to make sure you understand the concept. Well, that's the whole point. Your examples should not be gamified. They should touch upon a real world problem. That's very important. Okay? I I've mentioned this a couple of times. Please make sure you're not missing that part. Okay? Again, refer to the page. Okay? This this page is your final verdict for anything, right? Uh I do also recommend because you're getting all these questions and I do understand

the page is a bit deep, right? There are too many details on the page. So one of the things you can do is again utilize AI to the best just just copy it all put it in a put it in a chat bot put it in a chat GPD or whatever and just ask questions whatever are coming to your mind so that you get a clarity on what is okay and what is not okay right please please don't miss these important things out right it has to be a real world task you can see over here must simulate a real world task not games or

toys okay ki and anyone else who had a similar doubt I hope This is now clear. Yes. For the examples, we have already shared a page with you. I'll I'll probably share it one more time. But uh you you can take a look at that through the uh hugging face page that Ben shared in the beginning itself. Right. Let let me see if I can share that page with you guys. That's a great place to you know sort of rely on for real world examples. Okay. So, let's see. I think you can refer to this page. Let

Let me share another page with you. One second. Okay, I I'll give you a walk through of some real world uh environments. Just allow me a minute. Let me let me open that for you guys. Okay. So, this page is based on a previous hackathon that happened, right? And you can you can get a sense of the kind of things that people have built over here. Okay. So just refer to this page. Uh we have provided this in the chat as well. Team if you could if you could confirm once if we have provided this. Yeah I can see it. It's it's

already there in the chat. So you can refer to this one. Yeah that's why we have shared this link with you guys already. So please just refer to it. It's there in the chat. It's there in the live link. So please refer to it. All right. So this is the project gallery. You can see the kind of submissions people have made over here. Right. Let me see if I can show you some more examples. Let me let me check. Yeah. So this is the environment hub, right? So once you're on the once you open this hugging face.open

Open page right once you once you land here and over here you can see uh there's a link to open environment hub so just click and this is where you can see a lot of different environments that people have built right but again bear in mind no plagiarism right of course if an environment exists we we know it does exist right so there's no point in sort of copypasting it so just make sure you're adding novelty to the ideas that you have on your mind. But this is where you can find a lot of different uh

environments. Okay, cool. Uh sir, healthtorm, I'm just calling this out one more time. Super important. Please don't miss this part. I've we've repeatedly been calling this out. Okay, you do not have to use the open AI API, right? We have called it out a couple of times. You can use the hugging face token itself along with the router and this is part of the script that I've been the inference script. It's it's mentioned over there and that's what you need to take care of like let me give

you a final walk through of this one one last time. Okay, we sort of called this a couple of times but make sure you don't miss this part out. You can see over here there's an API base URL. The whole idea is if you look at the script. You'll see over here there's a router. So you don't need the open AI key. Your hugging face token will take care of hitting the open AI within the hugging face environment. Right? So you don't have to worry about it. You do not have to create that key. Just the hugging

face token is enough. That's why in the mandatory section it's mentioned. Make sure you're using the hugging face token. just create that. There are some free credits available within it. That should be good enough for uh for you to be able to work, right? You don't need anything else. Please don't miss this part out. Right? Okay. I hope this is now clear to everyone. All right? So guys, I think I'll just take some time now to uh give some closing remarks, right? Uh firstly, I want to

quickly mention that uh build teams, right? This is going to be way more fun and way easier if you're solving this with other people. You can already see uh while the task itself while while open ENV simplifies the process of building these environments but building the environment by itself does require some creativity at your end, right? It does require some experimentation and things would be it'll be you'll be better off if you're in teams, right? So do form teams. It's going to be a learning

opportunity for you, your friends, your colleagues. So uh work with them to towards building this. Okay, that's that's the first thing that I want to call out and make sure you are following through the page. If required, use the dashboard landing page, copy the contents, push them into some chat, GPT, claude, whatever you like and ask questions. The questions you were asking us that that page has the answer to all those questions essentially, right? So, just uh ask those questions over there.

you'll get more clarity. Okay. One thing that kept coming up again and again, open AI key. You don't need that. We have called it out multiple times. Make sure you're using the hugging face token. That would take care of things. There's a router present in the inference script. It'll automatically hit uh whatever model you select. Okay. So that that's that's pretty much it. And uh do enjoy while you're building this. Keep in mind the deadline which is 8th of April. you won't be able to make

submissions after that. Also calling out multiple submissions are allowed. We will consider your most recent accurate submission. Right? So if you have made submissions that don't work and then later you make a submission that works. Of course we'll consider that. So you don't have to worry about it. And any number of submissions are allowed. So also feel free to you know uh make a push as many times as you want. We are totally fine with it. So yeah before we close uh just a quick call out. I think

Ben and team have done a fantastic work for enabling us to uh use open and also be able to use hugging face to deploy all these environments. So kudos to that. Um thanks to Scaler for enabling this hackathon. That's that's again super duper important and uh make sure you put in your submissions for uh uh for any sort of uh issues you face. You have the helper page over here. Just refer to it. Yeah, Scalar School of Technology is enabling this platform and hackathon. And any updates that are

there, you will be receiving them. Hopefully once you logged in you would have already noticed you were getting uh you know all the updates over WhatsApp over your email. So that's going to continue to happen. We'll keep reminding you about the deadlines. So keep an eye out for that. And yeah rest most of all please have fun while uh working on this task. All the people uh who were there on this particular live are all present on LinkedIn. You can follow us on LinkedIn, you can follow Ben, you can follow me and uh

yeah, we we would be looking forward to your submissions. All right guys, that would be all from my side. Thank you. Thank you so much for all your time and good luck to all of you.



## pre_submission_checklist.docx

HF Space deploys
Automated ping to the Space URL — must return 200 and respond to reset()
OpenEnv spec compliance
Validate openenv.yaml, typed models, step()/reset()/state() endpoints
Dockerfile builds
Automated docker build on the submitted repo
Baseline reproduces
Run the submitted inference script — must complete without error and produce scores
3+ tasks with graders
Enumerate tasks, run each grader, verify scores in 0.0–1.0 range
Mandatory Additional Instructions
Before submitting, ensure the following variables are defined in your environment configuration:
API_BASE_URL The API endpoint for the LLM.
MODEL_NAME The model identifier to use for inference.
HF_TOKEN Your Hugging Face / API key.
The inference script must be named `inference.py` and placed in the root directory of the project
Participants must use OpenAI Client for all LLM calls using above variables
Participants must emit structured stdout logs strictly following the [START], [STEP], and [END] format defined in the sample inference.py provided below. Any deviation in field names, ordering, or formatting will result in incorrect evaluation scoring. Refer to the Sample Inference Script for the complete format specification and examples.
Infra Restrictions
Runtime of inference script should be less than 20min 
Make sure your env and inference can run on a machine with vcpu=2, memory=8gb
Validator
Run the pre-submission validation script before submitting


## pre_validation script.docx

PRE-VALIDATION SCRIPT
#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Prerequisites:
#   - Docker:       https://docs.docker.com/get-docker/
#   - openenv-core: pip install openenv-core
#   - curl (usually pre-installed)
#
# Run:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- <ping_url> [repo_dir]
#
#   Or download and run locally:
#     chmod +x validate-submission.sh
#     ./validate-submission.sh <ping_url> [repo_dir]
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   Path to your repo (default: current directory)
#
# Examples:
#   ./validate-submission.sh https://my-team.hf.space
#   ./validate-submission.sh https://my-team.hf.space ./my-repo
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)\n"
  printf "  repo_dir   Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi
PING_URL="${PING_URL%/}"
export PING_URL
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PING_URL/reset) ..."

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check your network connection and that the Space is running."
  hint "Try: curl -s -o /dev/null -w '%%{http_code}' -X POST $PING_URL/reset"
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sure your Space is running and the URL is correct."
  hint "Try opening $PING_URL in your browser first."
  stop_at "Step 1"
fi

log "${BOLD}Step 2/3: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0

## sample_inference_script.docx

"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 rewards=0.00,0.00,1.00
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from my_env_v4 import MyEnvV4Action, MyEnvV4Env
IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# Max possible reward: each token contributes 0.1, across all steps
_MAX_REWARD_PER_STEP = MAX_TOKENS * 0.1
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are interacting with a simple echo environment.
    Each turn you must send a message. The environment will echo it back.
    Reward is proportional to message length: reward = len(message) * 0.1
    Your goal is to maximize total reward by sending meaningful, substantive messages.
    Reply with exactly one message string — no quotes, no prefixes, just the message text.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last echoed message: {last_echoed!r}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Send your next message.
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, last_echoed, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "hello"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "hello"


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset() # OpenENV.reset()
        last_echoed = result.observation.echoed_message
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            message = get_model_message(client, step, last_echoed, last_reward, history)

            result = await env.step(MyEnvV4Action(message=message))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_echoed = obs.echoed_message
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())

## spec_v1.5.docx

╔══════════════════════════════════════════════════════════════════╗
║           MODGUARD-RL ENVIRONMENT SPECIFICATION                  ║
║           SPEC_VERSION 1.5.0                                     ║
╚══════════════════════════════════════════════════════════════════╝

SEMANTIC VERSIONING POLICY
  PATCH = wording only
  MINOR = new trigger, new band, new example
  MAJOR = table values, weights, GT set, or action set

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. CORE DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GT ∈ { approve, remove, legal_hold }
  escalate is never a ground truth label
  GT represents final disposition only

ACTIONS ∈ { approve, remove, escalate, legal_hold }

BUDGET
  escalation_budget = 1 per episode (only budget in system)
  legal_budget: does not exist
  budget_violated predicate: action == escalate AND escalation_budget == 0

DIFFICULTY LEVELS
  easy   → ai_confidence_score matches reality, hint always correct,
            English content, unambiguous violation
  medium → ai_confidence_score has ±0.2 noise, hint correct 60%,
            non-English possible, platform context matters
  hard   → ai_confidence_score manipulated, hint wrong 70%,
            coordinated campaigns, reviewer is adversarial

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. OBSERVATION (10 fields, exact, agent-visible)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  content_category        ContentCategory enum (see section 7)
  risk_level              low / medium / high / critical
  platform_context        public_post / private_group /
                          marketplace / messenger
  ai_confidence_score     float 0.0–1.0
  human_reviewer_hint     Action enum, Optional (None in hard mode)
  queue_pressure          int 1–5 (observable workload proxy,
                          reliability is inferred not read directly)
  reviewer_overturn_rate  float 0.0–1.0, Optional
                          None at step 1
                          noisy ±0.15 at step 2 and step 3
  step_number             int 1 / 2 / 3
  case_history            CaseHistory nested Pydantic model
                          { prior_escalations: int,
                            account_risk: float 0.0–1.0 }
  stage                   initial_review / escalation_review /
                          legal_review

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. STATE (internal, agent never sees)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  step_number             int
  stage                   StageEnum
  escalation_budget       int (starts 1)
  ground_truth            GTEnum (hidden)
  path_penalty_incurred   bool (sticky, False→True, never resets)
  budget_violated         bool
  action_history          list[Action]
  episode_done            bool

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. STATE MACHINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 (stage = initial_review)
  approve    → TERMINAL, no penalty
  remove     → TERMINAL, no penalty
  escalate   → STEP 2, stage = escalation_review,
               escalation_budget -= 1
  legal_hold → STEP 2, stage = legal_review, no budget consumed

STEP 2 (stage = escalation_review)
  approve    → TERMINAL, no penalty
  remove     → TERMINAL, no penalty
  escalate   → TERMINAL, step penalty -0.4, TRIGGER 1
  legal_hold → STEP 3 if risk == critical, no penalty
               TERMINAL if risk != critical AND GT == legal_hold,
                 no penalty
               TERMINAL if risk != critical AND GT != legal_hold,
                 step penalty -0.4, TRIGGER 2

STEP 2 (stage = legal_review)
  approve    → TERMINAL, no penalty
  remove     → TERMINAL, no penalty
  escalate   → TERMINAL, step penalty -0.4, TRIGGER 3
  legal_hold → TERMINAL if GT == legal_hold, no penalty
               TERMINAL if GT != legal_hold,
                 step penalty -0.4, TRIGGER 4

STEP 3 (stage = legal_review, risk == critical only)
  any action → TERMINAL
  budgets:     ignored (escalation_budget already spent)
  step reward: correctness-based penalty applies normally
  TRIGGER 5:   fires if step-1 action was legal_hold

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. PATH_PENALTY_INCURRED — SINGLE CANONICAL TRIGGER LIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  TRIGGER 1  stage=escalation_review, action=escalate, any condition
  TRIGGER 2  stage=escalation_review, action=legal_hold,
             risk != critical AND GT != legal_hold
  TRIGGER 3  stage=legal_review, action=escalate, any condition
  TRIGGER 4  stage=legal_review, action=legal_hold,
             GT != legal_hold
  TRIGGER 5  step 3 entered, step-1 action was legal_hold
             (wrong route into critical path)

  flag is sticky: set on transition, never resets
  read from final State snapshot inside grade()
  set as soon as triggering transition occurs, not at episode end

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. OPTIMAL PATHS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  definition: minimum steps over all admissible (penalty-free,
              flag-free) paths for (GT, risk)

  GT=approve      any risk     path: approve              optimal=1
  GT=remove       any risk     path: remove               optimal=1
  GT=legal_hold   non-critical path: legal_hold           optimal=1
                               also valid (suboptimal):
                               escalate → legal_hold      length=2
  GT=legal_hold   critical     path: escalate →
                               legal_hold → legal_hold    optimal=3

  note: escalate → legal_hold for GT=legal_hold non-critical
        is admissible (no penalty, no flag) but suboptimal
        process_score reflects 2 vs optimal 1 → 0.6

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. CORRECTNESS TABLE [GT][terminal_action]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  definition: correctness_score = table[GT][terminal_action]
  terminal_action = action taken at the step where episode ends

              approve   remove   escalate   legal_hold
  approve       1.0      0.5       0.1         0.0
  remove        0.5      1.0       0.3         0.1
  legal_hold    0.0      0.1       0.5         1.0

  calibration rationale:
  legal_hold + escalate = 0.5  agent recognised severity, stopped short
  legal_hold + remove   = 0.1  removal destroys evidence, worse than escalate
  legal_hold + approve  = 0.0  worst outcome, approved content needing legal hold
  remove     + escalate = 0.3  unnecessary escalation wastes resources
  approve    + escalate = 0.1  false positive, wastes reviewer time

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. GRADE FORMULA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  grade = correctness × 0.45
        + process     × 0.25
        + hint        × 0.20
        + speed       × 0.10

  weights sum: 0.45 + 0.25 + 0.20 + 0.10 = 1.00
  all subscores independently [0,1]
  grade naturally in [0,1], no clip needed

PROCESS SCORE — order of operations:
  1. budget_violated == True → 0.0, return immediately
  2. base = matrix[actual][optimal] using |actual - optimal|:

     actual \ optimal    1     2     3
     1                  1.0   0.6   0.2
     2                  0.6   1.0   0.6
     3                  0.2   0.6   1.0

  3. path_penalty_incurred == True → min(base, 0.4)
  4. return

  proof of completeness: max |actual - optimal| = |1-3| = 2
  all reachable (actual, optimal) pairs:
    (1,1), (2,1)           GT=approve/remove/legal_hold-nc
    (1,3), (2,3), (3,3)    GT=legal_hold-critical
  all six pairs defined in matrix, no episode undefined

HINT SCORE:
  evaluated at final terminal step only
  missing hint (None) → 0.5, neutral, documented
  followed  = terminal action == hint
  overrode  = terminal action != hint AND hint != GT (hint was wrong)
  ignored   = terminal action != hint AND hint == GT (hint was correct)

  wrong  + overrode  → 1.0   smart override
  correct + followed → 0.8   correctly trusted
  correct + ignored  → 0.3   unnecessary override
  wrong  + followed  → 0.0   blind trust, worst failure

SPEED SCORE:
  length 1 → 1.0
  length 2 → 0.6
  length 3 → 0.2
  length-only; routing quality lives in process score, not speed
  README note: "speed measures episode length only"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. ILLUSTRATIVE GRADES (all verified)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example A — optimal, correct, hint followed, length 1
  path: GT=approve, step1=approve
  correctness=1.0, process=1.0, hint=0.8, speed=1.0
  grade = 0.45 + 0.25 + 0.16 + 0.10 = 0.96

Example B — valid suboptimal, correct, hint followed, length 2
  path: GT=legal_hold nc, step1=escalate, step2=legal_hold
  no triggers, path_penalty_incurred=False
  correctness=1.0, process base=0.6 (|2-1|=1), no cap, hint=0.8, speed=0.6
  grade = 0.45 + 0.15 + 0.16 + 0.06 = 0.82

Example C — wrong route, flag set, hint overrode, length 2
  path: GT=legal_hold nc, step1=escalate, step2=escalate (TRIGGER 1)
  terminal action=escalate, correctness=table[legal_hold][escalate]=0.5
  process base=0.6 (|2-1|=1), capped to 0.4 (flag set)
  hint=1.0 (hint said approve, wrong, agent overrode), speed=0.6
  grade = 0.225 + 0.10 + 0.20 + 0.06 = 0.585

Example D — budget violated, wrong disposition, blind trust, length 2
  correctness=0.1, process=0.0 (budget short-circuit),
  hint=0.0 (blind trust), speed=0.6
  grade = 0.045 + 0.00 + 0.00 + 0.06 = 0.105

Easy mode typical spread: 0.82–0.96
Wrong-route correct-disposition max: ~0.85
  (1.0×0.45 + 0.4×0.25 + 1.0×0.20 + 1.0×0.10 = 0.85,
   only reachable if hint and speed are both at maximum)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10. CONTENT CATEGORIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  SEVERITY_CRITICAL         policy tier A (most serious)
  SEVERITY_HIGH             policy tier B
  SEVERITY_MEDIUM           policy tier C
  COORDINATED_INAUTHENTIC   platform integrity violation
  PLATFORM_INTEGRITY        ecosystem safety violation

  naming: neutral policy tiers, no graphic descriptions
  README describes these as "policy violation tiers" only

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
11. DEPLOYMENT AND CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  config filename:  openenv.yaml (confirmed from openenv init output)
  Dockerfile:       project root, not inside server/
  web interface:    ENV ENABLE_WEB_INTERFACE=true in Dockerfile
  deployment:       openenv push → Hugging Face Spaces
  inference LLM:    Qwen 72B Instruct via HF router, HF token only
  no LLM inside environment
  fully deterministic and synthetic
  single escalation_budget, no legal_budget

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
12. FILE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  modguard_rl/
  ├── Dockerfile
  ├── openenv.yaml
  ├── pyproject.toml
  ├── inference.py
  ├── README.md
  ├── server/
  │   ├── app.py
  │   ├── environment.py
  │   └── models.py
  └── client/
      └── client.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
13. README ONE-PARAGRAPH CONTRACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ModGuard-RL is a synthetic content-policy triage simulator.
  Each episode samples a difficulty tier, a hidden ground-truth
  disposition, and structured observation signals including content
  category, risk level, platform context, AI pre-moderation
  confidence, and a stochastic human reviewer recommendation.
  The agent chooses from four actions: approve, remove, escalate,
  or legal_hold. Episode length is variable — actions determine
  whether the episode branches to further review or terminates.
  Rewards combine ground-truth alignment, severity weighting,
  escalation costs, SLA penalties, and hint-handling quality.
  The grade function returns a documented 0–1 score with four
  weighted components so scores differ meaningfully across
  episodes and difficulty levels.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
END OF SPEC — SPEC_VERSION 1.5.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## openenv-course/README.md

# Building RL Environments with OpenEnv

A hands-on course for ML engineers, researchers, and hobbyists who want to use and build RL environments for LLM training.

**5 modules · ~45-60 min each · Markdown + Jupyter notebooks**

## Prerequisites

- Basic Python
- Familiarity with the Hugging Face ecosystem
- No RL experience required

## How to Use This Course

Each module has two parts:
1. **README.md** — Concepts, architecture, context. Read this first.
2. **notebook.ipynb** — Hands-on code. Open in Google Colab and run top-to-bottom.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## Modules

| # | Module | What You'll Learn | Notebook |
|---|--------|-------------------|----------|
| 1 | [Why OpenEnv?](module-1/README.md) | The RL loop, why Gym falls short, OpenEnv architecture | [Open →](module-1/notebook.ipynb) |
| 2 | [Using Existing Environments](module-2/README.md) | Environment Hub, type-safe models, policies, competition | [Open →](module-2/notebook.ipynb) |
| 3 | [Deploying Environments](module-3/README.md) | Local dev, Docker, HF Spaces, `openenv push` | [Open →](module-3/notebook.ipynb) |
| 4 | [Building Your Own Environment](module-4/README.md) | The 3-component pattern, scaffold → deploy | [Open →](module-4/notebook.ipynb) |
| 5 | [Training with OpenEnv + TRL](module-5/README.md) | GRPO, reward functions, Wordle training | [Open →](module-5/notebook.ipynb) |

## Quick Start

```bash
# Install OpenEnv core
pip install openenv-core

# Clone the OpenEnv repo to get typed environment clients
git clone https://github.com/meta-pytorch/OpenEnv.git
```

```python
import sys, os
repo = os.path.abspath('OpenEnv')
sys.path.insert(0, repo)
sys.path.insert(0, os.path.join(repo, 'src'))

# Echo environment — uses MCP tool-calling interface
from envs.echo_env import EchoEnv

with EchoEnv(base_url="https://openenv-echo-env.hf.space").sync() as env:
    env.reset()
    response = env.call_tool("echo_message", message="Hello, OpenEnv!")
    print(response)  # Hello, OpenEnv!

# OpenSpiel environments — use standard reset/step interface
from envs.openspiel_env import OpenSpielEnv
from envs.openspiel_env.models import OpenSpielAction

with OpenSpielEnv(base_url="https://openenv-openspiel-catch.hf.space").sync() as env:
    result = env.reset()
    result = env.step(OpenSpielAction(action_id=1, game_name="catch"))
    print(result.observation.legal_actions)
```

Every standard OpenEnv environment uses the same 3-method interface: `reset()`, `step()`, `state()`.

## Links

- [OpenEnv GitHub](https://github.com/meta-pytorch/OpenEnv)
- [Environment Hub Collection](https://huggingface.co/collections/openenv/environment-hub)
- [TRL Documentation](https://huggingface.co/docs/trl)

---

## Bonus: Scaling OpenEnv

For production workloads beyond a single container, see the scaling appendix below.

### WebSocket vs HTTP

OpenEnv uses WebSocket (`/ws`) for persistent sessions instead of stateless HTTP. Each `step()` call is a lightweight frame (~0.1ms overhead) over an existing connection, vs TCP handshake overhead (~10-50ms) with HTTP.

One container handles many isolated sessions — each WebSocket connection gets its own environment instance server-side.

![WebSocket vs HTTP](https://raw.githubusercontent.com/meta-pytorch/OpenEnv/main/tutorial/images/websocket.png)

### Single Container Scaling

Before adding containers, maximize a single deployment:

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKERS` | 4 | Uvicorn worker processes |
| `MAX_CONCURRENT_ENVS` | 100 | Max WebSocket sessions per worker |

With 8 workers, a single container can handle ~2,048 concurrent sessions for simple text environments.

### Multi-Container with Load Balancing

When a single container isn't enough, deploy multiple containers behind Envoy:

| Setup | Containers | Sessions/container | Total capacity |
|-------|------------|-------------------|----------------|
| Single | 1 | 100 | 100 |
| 4× containers | 4 | 100 | 400 |
| 8× containers | 8 | 100 | 800 |

### Benchmark Results

| Infrastructure | Max Concurrent (WS) | Cores | Sessions/Core |
|----------------|---------------------|-------|---------------|
| HF Spaces (free) | 128 | 2 | 64 |
| Local Uvicorn | 2,048 | 8 | 256 |
| Local Docker | 2,048 | 8 | 256 |
| SLURM multi-node | 16,384 | 96 | 171 |

![Scaling](https://raw.githubusercontent.com/meta-pytorch/OpenEnv/main/tutorial/images/scaling.png)

For full scaling experiments and code, see [burtenshaw/openenv-scaling](https://github.com/burtenshaw/openenv-scaling).

### Recommendations

- **Development / moderate load (<2K concurrent):** Single Uvicorn or Docker container. Best per-core efficiency (256 sessions/core).
- **Demos and published environments:** HF Spaces free tier, reliable up to 128 concurrent sessions.
- **Large-scale training (>2K concurrent):** Multi-node with Envoy load balancer. See [tutorial/03-scaling.md](https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/03-scaling.md).


## openenv-course/requirements.txt

# OpenEnv Course — Python dependencies
#
# Install everything needed to run all 5 modules:
#   pip install -r requirements.txt
#
# GPU (A100 40GB) is only required for Module 5 (GRPO training).

# Core OpenEnv framework
openenv-core>=0.2.2

# Server-side dependencies (needed to run environments locally in Module 3)
fastapi>=0.104.0
uvicorn>=0.24.0
fastmcp>=3.0.0
pydantic>=2.0.0

# Module 5: LLM training
trl>=0.17.0
transformers>=4.40.0
datasets>=2.18.0
accelerate>=0.28.0
# vllm           # Uncomment for Module 5 (requires CUDA + Linux)
# bitsandbytes   # Uncomment for Module 5 quantisation

# Experiment tracking (Module 5)
trackio

# HF Hub utilities
huggingface-hub>=0.22.0


## openenv-course/module-1/notebook.ipynb

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Module 1: Why OpenEnv? — Your First Environments\n",
    "\n",
    "In this notebook you'll connect to three real hosted OpenEnv environments and interact with each using the same 3-method interface: `reset()`, `step()`, `state()`.\n",
    "\n",
    "**Time:** ~15 min · **Difficulty:** Beginner · **GPU:** Not required"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "!pip install -q openenv-core fastmcp\n!git clone --depth=1 -q https://github.com/meta-pytorch/OpenEnv.git 2>/dev/null || true\n\nimport sys, os\nrepo = os.path.abspath('OpenEnv')\nfor p in [repo, os.path.join(repo, 'src')]:\n    if p not in sys.path:\n        sys.path.insert(0, p)\nprint(\"Setup complete!\")"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. The Echo Environment\n",
    "\n",
    "The simplest possible OpenEnv environment — it echoes back whatever you send. Perfect for learning the interface.\n",
    "\n",
    "Hosted at: `https://openenv-echo-env.hf.space`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from envs.echo_env import EchoEnv\n",
    "\n",
    "# EchoEnv extends MCPToolClient — it exposes tools, not raw reset/step actions.\n",
    "# MCP methods (list_tools, call_tool) are async; .sync() wraps them automatically\n",
    "# via SyncEnvClient.__getattr__, so the same .sync() pattern works here.\n",
    "with EchoEnv(base_url='https://openenv-echo-env.hf.space').sync() as env:\n",
    "    # reset() starts a new episode\n",
    "    result = env.reset()\n",
    "    print('After reset:')\n",
    "    print(f'  Observation: {result.observation}')\n",
    "    print(f'  Done: {result.done}')\n",
    "    print()\n",
    "\n",
    "    # Discover available tools\n",
    "    tools = env.list_tools()\n",
    "    print('Available tools:')\n",
    "    for tool in tools:\n",
    "        print(f'  - {tool.name}: {tool.description}')\n",
    "    print()\n",
    "\n",
    "    # call_tool() sends a message and returns the result\n",
    "    response = env.call_tool('echo_message', message='Hello, OpenEnv!')\n",
    "    print(f'echo_message(\"Hello, OpenEnv!\") -> {response}')\n",
    "\n",
    "    response = env.call_tool('echo_with_length', message='OpenEnv')\n",
    "    print(f'echo_with_length(\"OpenEnv\") -> {response}')\n",
    "\n",
    "    # state() returns episode metadata\n",
    "    state = env.state()\n",
    "    print(f'\\nState: step_count={state.step_count}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Three methods. That's the entire API. Every OpenEnv environment works exactly like this."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. OpenSpiel Catch\n",
    "\n",
    "Now let's connect to a real game. Catch is a simple single-player game from DeepMind's OpenSpiel:\n",
    "\n",
    "- A ball falls from the top of a 10×5 grid\n",
    "- You move a paddle left/right to catch it\n",
    "- Actions: `0` = left, `1` = stay, `2` = right\n",
    "- Reward: `+1` if caught, `0` if missed\n",
    "\n",
    "Same 3 methods, completely different game."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from envs.openspiel_env import OpenSpielEnv\n",
    "from envs.openspiel_env.models import OpenSpielAction\n",
    "\n",
    "OPENSPIEL_URL = 'https://openenv-openspiel-catch.hf.space'\n",
    "\n",
    "with OpenSpielEnv(base_url=OPENSPIEL_URL).sync() as env:\n",
    "    result = env.reset()\n",
    "    print('Game: Catch')\n",
    "    print(f'Legal actions: {result.observation.legal_actions}')\n",
    "    print(f'Info state length: {len(result.observation.info_state)}')\n",
    "    print()\n",
    "\n",
    "    # Play a few steps with a random policy\n",
    "    import random\n",
    "    step = 0\n",
    "    while not result.done:\n",
    "        action_id = random.choice(result.observation.legal_actions)\n",
    "        action_name = {0: 'LEFT', 1: 'STAY', 2: 'RIGHT'}[action_id]\n",
    "        result = env.step(OpenSpielAction(\n",
    "            action_id=action_id,\n",
    "            game_name='catch'\n",
    "        ))\n",
    "        step += 1\n",
    "        print(f'Step {step}: {action_name} -> reward={result.reward}, done={result.done}')\n",
    "\n",
    "    print(f'\\nFinal reward: {result.reward}')\n",
    "    state = env.state()\n",
    "    print(f'State: step_count={state.step_count}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Same pattern: `reset()` → `step()` → check `done`. The observation type is different (`OpenSpielObservation` vs `EchoObservation`), but the interface is identical."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. TextArena Wordle\n",
    "\n",
    "TextArena is a text-based game environment. Wordle gives you 6 attempts to guess a 5-letter word, with color-coded feedback after each guess.\n",
    "\n",
    "Hosted at: `https://burtenshaw-textarena.hf.space`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from envs.textarena_env import TextArenaEnv\n",
    "from envs.textarena_env.models import TextArenaAction\n",
    "\n",
    "TEXTARENA_URL = 'https://burtenshaw-textarena.hf.space'\n",
    "\n",
    "with TextArenaEnv(base_url=TEXTARENA_URL).sync() as env:\n",
    "    result = env.reset()\n",
    "    print('Wordle prompt:')\n",
    "    print(result.observation.prompt)\n",
    "    print()\n",
    "\n",
    "    # Make a few guesses\n",
    "    guesses = ['crane', 'slate', 'blind']\n",
    "    for guess in guesses:\n",
    "        if result.done:\n",
    "            break\n",
    "        result = env.step(TextArenaAction(message=f'[{guess}]'))\n",
    "        print(f'Guess: {guess}')\n",
    "        for msg in result.observation.messages:\n",
    "            print(f'  [{msg.category}] {msg.content}')\n",
    "        print(f'  Reward: {result.reward}, Done: {result.done}')\n",
    "        print()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Async vs Sync\n",
    "\n",
    "OpenEnv clients are async by default. For notebooks and simple scripts, use the `.sync()` wrapper:\n",
    "\n",
    "```python\n",
    "# Sync (notebooks, simple scripts)\n",
    "with EchoEnv(base_url=url).sync() as env:\n",
    "    result = env.reset()\n",
    "\n",
    "# Async (production, training loops)\n",
    "async with EchoEnv(base_url=url) as env:\n",
    "    result = await env.reset()\n",
    "```\n",
    "\n",
    "For this course, we'll use `.sync()` everywhere for simplicity."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Summary\n",
    "\n",
    "You connected to three completely different environments — Echo, Catch, Wordle — using the same interface:\n",
    "\n",
    "| Method | What it does |\n",
    "|--------|--------------|\n",
    "| `reset()` | Start a new episode |\n",
    "| `step(action)` | Take an action, get observation + reward |\n",
    "| `state()` | Get episode metadata |\n",
    "\n",
    "The action and observation types change per environment, but the pattern never does.\n",
    "\n",
    "**Next:** [Module 2](../module-2/README.md) — Using existing environments to build and compare policies."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

## openenv-course/module-1/README.md

# Module 1: Why OpenEnv? From Cartpole to Production RL

## The RL Loop in 60 Seconds

Reinforcement Learning is a loop:

```python
while not done:
    observation = environment.observe()
    action = policy.choose(observation)
    reward = environment.step(action)
    policy.learn(reward)
```

Observe → Act → Reward → Repeat. That's it.

The agent interacts with an environment, gets feedback, and improves. Every RL system — from game-playing bots to LLM fine-tuning with GRPO — follows this pattern.

## Why Gym/Gymnasium Falls Short for LLM Training

OpenAI Gym (now Gymnasium) is the standard for RL research. It works great for Cartpole. But when you try to use it for production LLM training, problems appear:

| Challenge | Gymnasium | What you actually need |
|-----------|-----------|----------------------|
| **Type Safety** | `obs[0][3]` — what is this? | `obs.info_state` — IDE knows |
| **Isolation** | Same process (can crash training) | Docker containers (fully isolated) |
| **Deployment** | "Works on my machine" | Same container everywhere |
| **Scaling** | Hard to distribute | Deploy to Kubernetes |
| **Language** | Python only | Any language via HTTP |
| **Debugging** | Cryptic numpy errors | Clear type errors |

The core issue: Gymnasium assumes your environment runs in the same process as your training code. That's fine for research. It's a disaster for production.

## The OpenEnv Philosophy

**RL environments should be microservices.**

You don't run your database in the same process as your web server. Same principle applies to RL environments:

- **Isolated** — Run in containers. Security + stability.
- **Standard** — HTTP/WebSocket API. Works from any language.
- **Versioned** — Docker images. Reproducible everywhere.
- **Scalable** — Deploy to cloud with one command.
- **Type-safe** — Catch bugs before they happen.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  YOUR TRAINING CODE                                        │
│                                                            │
│  env = EchoEnv(base_url="https://...")                    │
│  result = env.reset()           ← Type-safe!              │
│  result = env.step(action)      ← Type-safe!              │
│                                                            │
└─────────────────┬──────────────────────────────────────────┘
                  │
                  │  WebSocket / HTTP  (Language-Agnostic)
                  │
┌─────────────────▼──────────────────────────────────────────┐
│  DOCKER CONTAINER (HF Space, local, cloud)                 │
│                                                            │
│  ┌──────────────────────────────────────────────┐         │
│  │  FastAPI Server                              │         │
│  │  └─ Environment (reset, step, state)         │         │
│  │     └─ Your Game/Simulation Logic            │         │
│  └──────────────────────────────────────────────┘         │
│                                                            │
│  Isolated • Reproducible • Secure                          │
└────────────────────────────────────────────────────────────┘
```

The client uses the `/ws` WebSocket endpoint by default. You never see the HTTP details — just clean Python methods:

```python
env.reset()    # Under the hood: WebSocket message
env.step(...)  # Under the hood: WebSocket message
env.state()    # Under the hood: WebSocket message
```

## The 3-Method Interface

Every OpenEnv environment exposes exactly three methods:

| Method | What it does | Returns |
|--------|-------------|---------|
| `reset()` | Start a new episode | `StepResult` (observation, reward, done) |
| `step(action)` | Take an action | `StepResult` (observation, reward, done) |
| `state()` | Get episode metadata | `State` (episode_id, step_count, etc.) |

This is the same whether you're playing Catch, Wordle, Tic-Tac-Toe, or a custom environment you built yourself.

## The 3-Component Pattern

Every OpenEnv environment has three components:

```
my_env/
├── models.py              ← Type-safe contracts (Action, Observation, State)
├── client.py              ← What you import in training code
└── server/
    ├── environment.py     ← Game/simulation logic
    ├── app.py             ← FastAPI server
    └── Dockerfile         ← Container definition
```

**Server side** (runs in Docker):
```python
class Environment(ABC):
    def reset(self) -> Observation: ...
    def step(self, action: Action) -> Observation: ...
    @property
    def state(self) -> State: ...
```

**Client side** (your training code):
```python
class EnvClient(ABC):
    async def reset(self, **kwargs) -> StepResult: ...
    async def step(self, action) -> StepResult: ...
    async def state(self) -> State: ...
    def sync(self) -> SyncEnvClient: ...  # Sync wrapper for notebooks/scripts
```

Same interface on both sides. Communication via WebSocket. You focus on RL.

For simple MCP-based environments (like the Echo environment), the interface is
tool-based instead: `env.list_tools()` and `env.call_tool(name, **kwargs)`.

## What's Next

In the [notebook](notebook.ipynb), you'll connect to three real hosted environments — Echo, OpenSpiel Catch, and TextArena Wordle — and interact with each using the same pattern.

**Key takeaway:** Every OpenEnv environment has the same 3-method interface. Once you know one, you know them all.


## openenv-course/module-2/notebook.ipynb

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Module 2: Policy Competition on OpenSpiel\n",
    "\n",
    "Build 4 policies, compete them on Catch, then switch to another game with the same code.\n",
    "\n",
    "**Time:** ~20 min · **Difficulty:** Beginner · **GPU:** Not required"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "!pip install -q openenv-core\n!git clone --depth=1 -q https://github.com/meta-pytorch/OpenEnv.git 2>/dev/null || true\n\nimport sys, os\nrepo = os.path.abspath('OpenEnv')\nfor p in [repo, os.path.join(repo, 'src')]:\n    if p not in sys.path:\n        sys.path.insert(0, p)\nprint(\"Setup complete!\")"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Connect to Catch\n",
    "\n",
    "Catch: a ball falls from the top of a 10×5 grid. Move your paddle to catch it.\n",
    "\n",
    "- Actions: `0` = LEFT, `1` = STAY, `2` = RIGHT\n",
    "- Reward: `+1` if caught, `0` if missed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from envs.openspiel_env import OpenSpielEnv\n",
    "from envs.openspiel_env.models import OpenSpielAction, OpenSpielObservation\n",
    "import random\n",
    "\n",
    "CATCH_URL = 'https://openenv-openspiel-catch.hf.space'\n",
    "\n",
    "# Quick sanity check\n",
    "with OpenSpielEnv(base_url=CATCH_URL).sync() as env:\n",
    "    result = env.reset()\n",
    "    print(f'Legal actions: {result.observation.legal_actions}')\n",
    "    print(f'Info state shape: {len(result.observation.info_state)} values')\n",
    "    print(f'Game phase: {result.observation.game_phase}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Define Four Policies\n",
    "\n",
    "Each policy takes an `OpenSpielObservation` and returns an action ID."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class RandomPolicy:\n",
    "    \"\"\"Pure random — baseline.\"\"\"\n",
    "    name = \"Random\"\n",
    "\n",
    "    def select_action(self, obs: OpenSpielObservation) -> int:\n",
    "        return random.choice(obs.legal_actions)\n",
    "\n",
    "\n",
    "class AlwaysStayPolicy:\n",
    "    \"\"\"Never moves — hopes ball lands on paddle.\"\"\"\n",
    "    name = \"Always Stay\"\n",
    "\n",
    "    def select_action(self, obs: OpenSpielObservation) -> int:\n",
    "        return 1  # STAY\n",
    "\n",
    "\n",
    "class SmartPolicy:\n",
    "    \"\"\"Moves paddle toward ball — optimal for Catch.\"\"\"\n",
    "    name = \"Smart Heuristic\"\n",
    "\n",
    "    def select_action(self, obs: OpenSpielObservation) -> int:\n",
    "        info_state = obs.info_state\n",
    "        grid_width = 5\n",
    "\n",
    "        # Find ball column (first 1.0 in the flattened grid)\n",
    "        ball_col = None\n",
    "        for idx, val in enumerate(info_state):\n",
    "            if abs(val - 1.0) < 0.01:\n",
    "                ball_col = idx % grid_width\n",
    "                break\n",
    "\n",
    "        # Paddle is in the last row\n",
    "        last_row = info_state[-grid_width:]\n",
    "        paddle_col = last_row.index(1.0)\n",
    "\n",
    "        if ball_col is not None:\n",
    "            if paddle_col < ball_col:\n",
    "                return 2  # RIGHT\n",
    "            elif paddle_col > ball_col:\n",
    "                return 0  # LEFT\n",
    "        return 1  # STAY\n",
    "\n",
    "\n",
    "class LearningPolicy:\n",
    "    \"\"\"Epsilon-greedy — starts random, learns to be smart.\"\"\"\n",
    "    name = \"Epsilon-Greedy\"\n",
    "\n",
    "    def __init__(self):\n",
    "        self.steps = 0\n",
    "        self._smart = SmartPolicy()\n",
    "\n",
    "    def select_action(self, obs: OpenSpielObservation) -> int:\n",
    "        self.steps += 1\n",
    "        epsilon = max(0.1, 1.0 - self.steps / 100)\n",
    "        if random.random() < epsilon:\n",
    "            return random.choice(obs.legal_actions)\n",
    "        return self._smart.select_action(obs)\n",
    "\n",
    "\n",
    "print(\"Policies defined: Random, Always Stay, Smart Heuristic, Epsilon-Greedy\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Run a Single Episode\n",
    "\n",
    "Helper to play one full game and return whether the ball was caught."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import here so run_episode is self-contained even if cell[3] is skipped\n",
    "from envs.openspiel_env.models import OpenSpielAction\n",
    "\n",
    "def run_episode(env, policy, verbose=False):\n",
    "    \"\"\"Play one episode. Returns 1 if caught, 0 if missed.\"\"\"\n",
    "    result = env.reset()\n",
    "    step = 0\n",
    "\n",
    "    while not result.done:\n",
    "        action_id = policy.select_action(result.observation)\n",
    "        if verbose:\n",
    "            name = {0: 'LEFT', 1: 'STAY', 2: 'RIGHT'}.get(action_id, str(action_id))\n",
    "            print(f'  Step {step}: {name}')\n",
    "        result = env.step(OpenSpielAction(action_id=action_id, game_name='catch'))\n",
    "        step += 1\n",
    "\n",
    "    caught = 1 if result.reward and result.reward > 0 else 0\n",
    "    if verbose:\n",
    "        status = 'Caught!' if caught else 'Missed'\n",
    "        print(f'  Result: {status} (reward={result.reward})')\n",
    "    return caught\n",
    "\n",
    "\n",
    "# Demo: one verbose episode with SmartPolicy\n",
    "with OpenSpielEnv(base_url=CATCH_URL).sync() as env:\n",
    "    print('Smart policy — single episode:')\n",
    "    run_episode(env, SmartPolicy(), verbose=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Policy Competition\n",
    "\n",
    "Run 50 episodes per policy and compare success rates."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "NUM_EPISODES = 50\n",
    "\n",
    "policies = [\n",
    "    RandomPolicy(),\n",
    "    AlwaysStayPolicy(),\n",
    "    SmartPolicy(),\n",
    "    LearningPolicy(),\n",
    "]\n",
    "\n",
    "results = {}\n",
    "\n",
    "with OpenSpielEnv(base_url=CATCH_URL).sync() as env:\n",
    "    for policy in policies:\n",
    "        wins = sum(run_episode(env, policy) for _ in range(NUM_EPISODES))\n",
    "        rate = wins / NUM_EPISODES * 100\n",
    "        results[policy.name] = rate\n",
    "        print(f\"{policy.name:20s} — {rate:5.1f}% ({wins}/{NUM_EPISODES})\")\n",
    "\n",
    "print(\"\\n--- Results ---\")\n",
    "for name, rate in sorted(results.items(), key=lambda x: -x[1]):\n",
    "    bar = \"█\" * int(rate / 2)\n",
    "    print(f\"{name:20s} [{bar:<50}] {rate:.1f}%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Expected results:\n",
    "- **Random**: ~20% (pure luck)\n",
    "- **Always Stay**: ~20% (terrible strategy)\n",
    "- **Smart Heuristic**: ~100% (optimal)\n",
    "- **Epsilon-Greedy**: ~80-90% (improves over episodes)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Switch Games\n",
    "\n",
    "The same `OpenSpielEnv` client works for all 6 OpenSpiel games. Let's try Tic-Tac-Toe — the observation format is identical, only the game logic changes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "TTT_URL = \"https://openenv-openspiel-tictactoe.hf.space\"\n",
    "\n",
    "with OpenSpielEnv(base_url=TTT_URL).sync() as env:\n",
    "    result = env.reset()\n",
    "    print(f\"Game: Tic-Tac-Toe\")\n",
    "    print(f\"Legal actions: {result.observation.legal_actions}\")\n",
    "    print(f\"Info state: {result.observation.info_state}\")\n",
    "    print(f\"Current player: {result.observation.current_player_id}\")\n",
    "    print()\n",
    "\n",
    "    # Play randomly until game ends\n",
    "    step = 0\n",
    "    while not result.done:\n",
    "        action_id = random.choice(result.observation.legal_actions)\n",
    "        result = env.step(OpenSpielAction(action_id=action_id, game_name=\"tic_tac_toe\"))\n",
    "        step += 1\n",
    "        print(f\"Step {step}: action={action_id}, reward={result.reward}, done={result.done}\")\n",
    "\n",
    "    print(f\"\\nGame over! Final reward: {result.reward}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Same client class. Same observation type. Different game. That's the OpenEnv promise.\n",
    "\n",
    "## Summary\n",
    "\n",
    "- Built 4 policies with increasing sophistication\n",
    "- Ran a 50-episode competition on Catch\n",
    "- Switched to Tic-Tac-Toe with zero code changes to the client\n",
    "\n",
    "All policies work with `OpenSpielObservation` — you read `info_state`, `legal_actions`, and `done`. The game logic is on the server. Your code is on the client.\n",
    "\n",
    "**Next:** [Module 3](../module-3/README.md) — Deploying environments to HF Spaces."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

## openenv-course/module-2/README.md

# Module 2: Using Existing Environments

## The Environment Hub

OpenEnv environments live on Hugging Face Spaces. The [Environment Hub collection](https://huggingface.co/collections/openenv/environment-hub) has ready-to-use environments you can connect to immediately.

Every Space gives you three things:

| Component | What it provides | How to access |
|-----------|------------------|---------------|
| **Server** | Running environment endpoint | `https://<username>-<space-name>.hf.space` |
| **Repository** | Installable Python package | `pip install git+https://huggingface.co/spaces/<space>` |
| **Registry** | Docker container image | `docker pull registry.hf.space/<space>:latest` |

You don't build environments to use them. Install the client, point it at a server, and go.

## Type-Safe Models

Every OpenEnv environment defines typed models for actions, observations, and state. These aren't just documentation — they're real Python dataclasses that your IDE can autocomplete and your type checker can validate.

For OpenSpiel environments (Pydantic models — `done` and `reward` are inherited from `Observation`):

```python
from openenv.core.env_server import Action, Observation, State
from pydantic import Field
from typing import Any, Dict, List, Optional

class OpenSpielAction(Action):
    action_id: int                              # Which action to take
    game_name: str = "catch"                   # Which game
    game_params: Dict[str, Any] = Field(default_factory=dict)  # Game config

class OpenSpielObservation(Observation):
    # done: bool and reward: Optional[float] are inherited from Observation
    info_state: List[float]      # Game state as a vector
    legal_actions: List[int]     # Valid actions this step
    game_phase: str = "playing"  # Current phase
    current_player_id: int = 0   # Whose turn
    opponent_last_action: Optional[int] = None
```

No more guessing what `obs[0][3]` means.

## OpenSpiel Integration

OpenEnv wraps 6 games from DeepMind's OpenSpiel library:

| Single-Player | Multi-Player |
|---------------|-------------|
| Catch — catch falling ball | Tic-Tac-Toe — classic 3×3 |
| Cliff Walking — navigate grid | Kuhn Poker — imperfect info |
| 2048 — tile puzzle | |
| Blackjack — card game | |

All six use the same `OpenSpielEnv` client and the same `OpenSpielAction`/`OpenSpielObservation` types. The only difference is the `game_name` parameter.

## Writing Policies

A policy is just a function that takes an observation and returns an action. Here are four approaches for Catch:

**Random** — baseline, ~20% success:
```python
def random_policy(obs):
    return random.choice(obs.legal_actions)
```

**Always Stay** — terrible, ~20% success:
```python
def stay_policy(obs):
    return 1  # STAY
```

**Smart Heuristic** — optimal, 100% success:
```python
def smart_policy(obs):
    ball_col = find_ball(obs.info_state)
    paddle_col = find_paddle(obs.info_state)
    if paddle_col < ball_col: return 2  # RIGHT
    if paddle_col > ball_col: return 0  # LEFT
    return 1  # STAY
```

**Epsilon-Greedy** — learns over time, ~85% success:
```python
def learning_policy(obs, step):
    epsilon = max(0.1, 1.0 - step / 100)
    if random.random() < epsilon:
        return random.choice(obs.legal_actions)
    return smart_policy(obs)
```

The key insight: all four policies work with the same `OpenSpielObservation` type. Swap the game from Catch to Tic-Tac-Toe and the observation format stays the same — only the game logic changes.

## Switching Games

Because all OpenSpiel games share the same client interface, switching games is trivial:

```python
# Catch
with OpenSpielEnv(base_url="https://openenv-openspiel-catch.hf.space").sync() as env:
    result = env.reset()

# Tic-Tac-Toe — same client, different URL
with OpenSpielEnv(base_url="https://openenv-openspiel-tictactoe.hf.space").sync() as env:
    result = env.reset()
```

Your policy code doesn't change. The observation has the same fields. You just need a new strategy for the new game.

## What's Next

In the [notebook](notebook.ipynb), you'll build and compare 4 policies on Catch, run a competition, then switch to another game with the same client code.

**Key takeaway:** You don't build environments to use them. Same client interface across all games.


## openenv-course/module-3/notebook.ipynb

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Module 3: Clone, Modify, Deploy\n",
    "\n",
    "Clone the Echo environment from the OpenEnv repo, modify it, test locally, and deploy to HF Spaces.\n",
    "\n",
    "**Time:** ~25 min · **Difficulty:** Intermediate · **GPU:** Not required\n",
    "\n",
    "> **Note:** Deployment to HF Spaces (Step 6) requires a Hugging Face account and token.\n",
    "> All other steps run locally."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "!pip install -q openenv-core fastmcp fastapi uvicorn\n!git clone --depth=1 -q https://github.com/meta-pytorch/OpenEnv.git 2>/dev/null || true\n\nimport sys, os\nrepo = os.path.abspath('OpenEnv')\nfor p in [repo, os.path.join(repo, 'src')]:\n    if p not in sys.path:\n        sys.path.insert(0, p)\nprint(\"Setup complete!\")"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Verify the Hosted Echo Environment\n",
    "\n",
    "First, let's confirm the hosted Echo environment works."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from envs.echo_env import EchoEnv\n",
    "\n",
    "ECHO_URL = 'https://openenv-echo-env.hf.space'\n",
    "\n",
    "with EchoEnv(base_url=ECHO_URL).sync() as env:\n",
    "    result = env.reset()\n",
    "    response = env.call_tool('echo_message', message='ping')\n",
    "    print(f'Sent: ping')\n",
    "    print(f'Received: {response}')\n",
    "    print('The standard Echo returns exactly what you send.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Clone the Echo Environment\n",
    "\n",
    "Clone the Space repository to get the full source code."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Copy the echo_env from the cloned OpenEnv repo into a working directory\n",
    "import shutil, os\n",
    "\n",
    "src = os.path.join(os.path.abspath('OpenEnv'), 'envs', 'echo_env')\n",
    "dst = 'echo-env-modified'\n",
    "\n",
    "if os.path.exists(dst):\n",
    "    shutil.rmtree(dst)\n",
    "shutil.copytree(src, dst)\n",
    "\n",
    "# Ensure server/ is a proper Python package so uvicorn can import server.app\n",
    "# (relative imports inside app.py require a real package, not a namespace package)\n",
    "for pkg_dir in [dst, os.path.join(dst, 'server')]:\n",
    "    init_file = os.path.join(pkg_dir, '__init__.py')\n",
    "    if not os.path.exists(init_file):\n",
    "        open(init_file, 'w').close()\n",
    "\n",
    "print('Copied echo_env to echo-env-modified/')\n",
    "print('Created __init__.py files for proper package import')\n",
    "os.listdir(dst)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Explore the Structure\n",
    "\n",
    "Every OpenEnv environment follows the same layout."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import glob\n",
    "files = sorted(glob.glob('echo-env-modified/**/*', recursive=True))\n",
    "for f in files:\n",
    "    if os.path.isfile(f):\n",
    "        print(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Look at the MCP tool definitions in the echo environment\n",
    "env_file = 'echo-env-modified/server/echo_environment.py'\n",
    "with open(env_file) as f:\n",
    "    print(f.read())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Modify the Environment\n",
    "\n",
    "Let's make a \"Reverse Echo\" — instead of echoing back the message, it reverses it.\n",
    "\n",
    "We'll modify the `step()` method in `environment.py`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "env_file = 'echo-env-modified/server/echo_environment.py'\n",
    "\n",
    "with open(env_file) as f:\n",
    "    content = f.read()\n",
    "\n",
    "print('Original echo_environment.py:')\n",
    "print(content)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Modify: make echo_message reverse the input\n",
    "# The MCP tool currently returns `message`; we change it to `message[::-1]`\n",
    "\n",
    "modified = content.replace(\n",
    "    'return message',\n",
    "    'return message[::-1]',\n",
    "    1  # Replace only the first occurrence (in echo_message tool)\n",
    ")\n",
    "\n",
    "with open(env_file, 'w') as f:\n",
    "    f.write(modified)\n",
    "\n",
    "print('Modified echo_environment.py (echo_message now reverses the input):')\n",
    "# Show the relevant section\n",
    "for line in modified.split('\\n'):\n",
    "    if 'echo_message' in line or 'return' in line or '@mcp' in line:\n",
    "        print(f'  {line}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Test Locally\n",
    "\n",
    "Start the modified server and connect to it.\n",
    "\n",
    "> In Colab, we'll start the server as a background process. Locally, you'd run `uv run server` in a separate terminal."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import subprocess\n",
    "import time\n",
    "import sys\n",
    "import os\n",
    "\n",
    "# The server app imports from openenv (installed) and envs.echo_env (in OpenEnv repo).\n",
    "# We run from the echo-env-modified directory so its server/ is importable.\n",
    "env = os.environ.copy()\n",
    "env['PYTHONPATH'] = os.pathsep.join([\n",
    "    os.path.abspath('echo-env-modified'),\n",
    "    os.path.abspath('OpenEnv'),\n",
    "    os.path.abspath('OpenEnv/src'),\n",
    "] + env.get('PYTHONPATH', '').split(os.pathsep))\n",
    "\n",
    "server = subprocess.Popen(\n",
    "    [sys.executable, '-m', 'uvicorn', 'server.app:app',\n",
    "     '--host', '0.0.0.0', '--port', '8001'],\n",
    "    cwd='echo-env-modified',\n",
    "    env=env,\n",
    "    stdout=subprocess.PIPE,\n",
    "    stderr=subprocess.PIPE,\n",
    ")\n",
    "\n",
    "# Give it time to start\n",
    "time.sleep(4)\n",
    "print(f'Server started (PID: {server.pid})')\n",
    "\n",
    "# Check it's healthy\n",
    "import urllib.request\n",
    "try:\n",
    "    with urllib.request.urlopen('http://localhost:8001/health', timeout=5) as r:\n",
    "        print(f'Health: {r.read().decode()}')\n",
    "except Exception as e:\n",
    "    print(f'Health check failed: {e}')\n",
    "    # Print server stderr for debugging\n",
    "    err = server.stderr.read1(1024).decode(errors='replace')\n",
    "    if err:\n",
    "        print(f'Server stderr: {err}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test the modified environment\n",
    "# Since this is an MCP env, we use EchoEnv.call_tool()\n",
    "from envs.echo_env import EchoEnv\n",
    "\n",
    "with EchoEnv(base_url='http://localhost:8001').sync() as env:\n",
    "    result = env.reset()\n",
    "\n",
    "    test_messages = ['Hello', 'OpenEnv', 'Reverse this!']\n",
    "    for msg in test_messages:\n",
    "        response = env.call_tool('echo_message', message=msg)\n",
    "        print(f'Sent: {msg:20s} -> Received: {response}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clean up the server\n",
    "server.terminate()\n",
    "server.wait()\n",
    "print(\"Server stopped.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Deploy to HF Spaces\n",
    "\n",
    "Once your environment works locally, deploy it with `openenv push`.\n",
    "\n",
    "```bash\n",
    "cd echo-env-modified\n",
    "openenv push --repo-id YOUR_USERNAME/reverse-echo-env\n",
    "```\n",
    "\n",
    "Your environment is now live at:\n",
    "- **API:** `https://YOUR_USERNAME-reverse-echo-env.hf.space`\n",
    "- **Web UI:** `https://YOUR_USERNAME-reverse-echo-env.hf.space/web`\n",
    "- **Docs:** `https://YOUR_USERNAME-reverse-echo-env.hf.space/docs`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Uncomment and run to deploy (requires HF token)\n",
    "# !cd echo-env-modified && openenv push --repo-id YOUR_USERNAME/reverse-echo-env"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Connect to Your Deployed Environment\n",
    "\n",
    "After deployment, install the client and connect:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Uncomment after deploying\n",
    "# !pip install -q git+https://huggingface.co/spaces/YOUR_USERNAME/reverse-echo-env\n",
    "#\n",
    "# with EchoEnv(base_url=\"https://YOUR_USERNAME-reverse-echo-env.hf.space\").sync() as env:\n",
    "#     result = env.reset()\n",
    "#     result = env.step(EchoAction(message=\"Deployed!\"))\n",
    "#     print(f\"Response from your Space: {result.observation}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Docker Deployment (Alternative)\n",
    "\n",
    "You can also pull and run the Docker image locally:\n",
    "\n",
    "```bash\n",
    "# Pull from HF registry (after deploying)\n",
    "docker pull registry.hf.space/YOUR_USERNAME-reverse-echo-env:latest\n",
    "docker run -d -p 8001:8000 registry.hf.space/YOUR_USERNAME-reverse-echo-env:latest\n",
    "\n",
    "# Or build from source\n",
    "cd echo-env-modified\n",
    "docker build -t reverse-echo:latest -f server/Dockerfile .\n",
    "docker run -d -p 8001:8000 reverse-echo:latest\n",
    "```\n",
    "\n",
    "Connect the same way:\n",
    "```python\n",
    "with EchoEnv(base_url=\"http://localhost:8001\").sync() as env:\n",
    "    result = env.reset()\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Summary\n",
    "\n",
    "What you did:\n",
    "1. Cloned an existing environment from HF Spaces\n",
    "2. Explored its structure (models, client, server)\n",
    "3. Modified the environment logic (echo → reverse echo)\n",
    "4. Tested locally with uvicorn\n",
    "5. Deployed to HF Spaces with `openenv push`\n",
    "\n",
    "The workflow is always: **clone → modify → test → deploy**.\n",
    "\n",
    "**Next:** [Module 4](../module-4/README.md) — Building an environment from scratch."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

## openenv-course/module-3/README.md

# Module 3: Deploying Environments

## Three Things a Space Gives You

Every HF Space running an OpenEnv environment provides three access methods:

| Component | What it provides | How to access |
|-----------|------------------|---------------|
| **Server** | Running environment endpoint | `https://<username>-<space-name>.hf.space` |
| **Repository** | Pip-installable Python package | `pip install git+https://huggingface.co/spaces/<space>` |
| **Registry** | Docker container image | `docker pull registry.hf.space/<space>:latest` |

One deployment. Three ways to use it.

## Local Development with Uvicorn

The fastest iteration loop: clone a Space and run it locally.

```bash
# Clone from HF Space
git clone https://huggingface.co/spaces/openenv/echo-env
cd echo-env

# Install and run
uv sync
uv run server
```

Or with uvicorn directly:

```bash
uvicorn echo_env.server.app:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag restarts the server when you change code. Essential for development.

Test it:
```bash
curl http://localhost:8000/health
# {"status": "healthy"}
```

Connect from Python:
```python
with EchoEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
```

## Docker Deployment

Docker gives you isolation and reproducibility.

### Pull from a Space's registry:
```bash
docker pull registry.hf.space/openenv-echo-env:latest
docker run -d -p 8000:8000 registry.hf.space/openenv-echo-env:latest
```

### Build from source:
```bash
git clone https://huggingface.co/spaces/openenv/echo-env
cd echo-env
docker build -t my-echo-env:latest -f server/Dockerfile .
docker run -d -p 8000:8000 my-echo-env:latest
```

### With environment variables:
```bash
docker run -d -p 8000:8000 \
    -e WORKERS=4 \
    -e MAX_CONCURRENT_ENVS=100 \
    my-echo-env:latest
```

## Deploying to HF Spaces

### Using `openenv push`

The fastest path from local code to a running endpoint:

```bash
cd my_env
openenv push --repo-id username/my-env
```

Your environment is now live:
- **API endpoint:** `https://username-my-env.hf.space`
- **Web UI:** `https://username-my-env.hf.space/web`
- **API docs:** `https://username-my-env.hf.space/docs`
- **Health check:** `https://username-my-env.hf.space/health`

### The `openenv.yaml` Manifest

Controls Space settings:

```yaml
name: my_env
version: "1.0.0"
description: My custom environment
```

### Environment Variables

Configure via Space Settings → Variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKERS` | 4 | Uvicorn worker processes |
| `PORT` | 8000 | Server port |
| `HOST` | 0.0.0.0 | Bind address |
| `MAX_CONCURRENT_ENVS` | 100 | Max WebSocket sessions |

### Hardware Options

| Tier | vCPU | RAM | Cost |
|------|------|-----|------|
| CPU Basic (Free) | 2 | 16GB | Free |
| CPU Upgrade | 8 | 32GB | $0.03/hr |

Free tier handles ~128 concurrent sessions — enough for development and demos.

## The Full Workflow

```
1. openenv init my_env       # Scaffold
2. Edit server/environment.py # Implement logic
3. uv run server              # Test locally
4. openenv push               # Deploy to HF Spaces
5. pip install git+https://huggingface.co/spaces/username/my-env  # Install client
```

## Choosing Your Access Method

| Method | Use when | Pros | Cons |
|--------|----------|------|------|
| **Remote Space** | Quick testing, low volume | Zero setup | Network latency |
| **Local Docker** | Development, high throughput | Full control, no network | Requires Docker |
| **Local Uvicorn** | Fast iteration | Fastest reload | No isolation |

## What's Next

In the [notebook](notebook.ipynb), you'll clone the Echo environment, modify it, run it locally, and deploy your modified version to HF Spaces.

**Key takeaway:** One Space gives you a running server, a pip-installable package, and a Docker image. `openenv push` gets you there in one command.


## openenv-course/module-4/notebook.ipynb

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Module 4: Build a Word Game Environment\n",
    "\n",
    "Build a letter-guessing (Hangman-style) environment from scratch using the OpenEnv pattern.\n",
    "\n",
    "**Time:** ~30 min · **Difficulty:** Intermediate · **GPU:** Not required"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "!pip install -q openenv-core\n!git clone --depth=1 -q https://github.com/meta-pytorch/OpenEnv.git 2>/dev/null || true\n\nimport sys, os\nrepo = os.path.abspath('OpenEnv')\nfor p in [repo, os.path.join(repo, 'src')]:\n    if p not in sys.path:\n        sys.path.insert(0, p)\nprint(\"Setup complete!\")"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Define the Types\n",
    "\n",
    "Every OpenEnv environment starts with its data contracts: what actions can you take, what do you observe, what metadata exists?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from dataclasses import dataclass, field\n",
    "from typing import List, Optional, Dict, Any\n",
    "\n",
    "# These would normally go in models.py\n",
    "\n",
    "@dataclass\n",
    "class WordGameAction:\n",
    "    \"\"\"Player guesses a single letter.\"\"\"\n",
    "    guess: str\n",
    "    metadata: Dict[str, Any] = field(default_factory=dict)\n",
    "\n",
    "@dataclass\n",
    "class WordGameObservation:\n",
    "    \"\"\"What the player sees after each guess.\"\"\"\n",
    "    done: bool\n",
    "    reward: Optional[float]\n",
    "    masked_word: str            # e.g., \"p_th_n\"\n",
    "    guessed_letters: List[str]  # All letters tried\n",
    "    attempts_remaining: int\n",
    "    message: str                # Feedback text\n",
    "    metadata: Dict[str, Any] = field(default_factory=dict)\n",
    "\n",
    "@dataclass\n",
    "class WordGameState:\n",
    "    \"\"\"Episode metadata.\"\"\"\n",
    "    episode_id: Optional[str] = None\n",
    "    step_count: int = 0\n",
    "    target_word: str = \"\"\n",
    "    max_attempts: int = 6\n",
    "\n",
    "print(\"Types defined: WordGameAction, WordGameObservation, WordGameState\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Implement the Environment\n",
    "\n",
    "The environment implements three methods: `reset()`, `step()`, and `state`. This is where the game logic lives."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import random\nimport uuid\n\nWORDS = [\n    \"python\", \"neural\", \"tensor\", \"matrix\", \"vector\",\n    \"kernel\", \"lambda\", \"signal\", \"binary\", \"cipher\",\n    \"model\", \"layer\", \"epoch\", \"batch\", \"token\",\n]\n\nclass WordGameEnvironment:\n    \"\"\"A letter-guessing game environment following the OpenEnv pattern.\"\"\"\n\n    def __init__(self):\n        self._state = WordGameState()\n        self._target = \"\"\n        self._guessed = set()\n        self._remaining = 6\n\n    def reset(self) -> WordGameObservation:\n        \"\"\"Start a new episode with a random word.\"\"\"\n        self._target = random.choice(WORDS)\n        self._guessed = set()\n        self._remaining = 10\n        self._state = WordGameState(\n            episode_id=str(uuid.uuid4()),\n            step_count=0,\n            target_word=self._target,\n            max_attempts=10,\n        )\n        return WordGameObservation(\n            done=False,\n            reward=None,\n            masked_word=self._mask(),\n            guessed_letters=[],\n            attempts_remaining=self._remaining,\n            message=f\"Guess letters in a {len(self._target)}-letter word!\",\n        )\n\n    def step(self, action: WordGameAction) -> WordGameObservation:\n        \"\"\"Process a letter guess.\"\"\"\n        letter = action.guess.lower().strip()\n        self._state.step_count += 1\n\n        # Already guessed?\n        if letter in self._guessed:\n            return WordGameObservation(\n                done=False,\n                reward=0.0,\n                masked_word=self._mask(),\n                guessed_letters=sorted(self._guessed),\n                attempts_remaining=self._remaining,\n                message=f\"Already guessed '{letter}'. Try another.\",\n            )\n\n        self._guessed.add(letter)\n\n        if letter in self._target:\n            message = f\"'{letter}' is in the word!\"\n        else:\n            self._remaining -= 1\n            message = f\"'{letter}' is not in the word.\"\n\n        # Check win/lose\n        masked = self._mask()\n        won = \"_\" not in masked\n        lost = self._remaining <= 0\n        done = won or lost\n\n        if won:\n            reward = 1.0\n            message = f\"You got it! The word was '{self._target}'.\"\n        elif lost:\n            reward = 0.0\n            message = f\"Out of attempts. The word was '{self._target}'.\"\n        else:\n            reward = 0.0\n\n        return WordGameObservation(\n            done=done,\n            reward=reward,\n            masked_word=masked,\n            guessed_letters=sorted(self._guessed),\n            attempts_remaining=self._remaining,\n            message=message,\n        )\n\n    @property\n    def state(self) -> WordGameState:\n        return self._state\n\n    def _mask(self) -> str:\n        \"\"\"Show guessed letters, hide the rest.\"\"\"\n        return \"\".join(c if c in self._guessed else \"_\" for c in self._target)\n\nprint(\"WordGameEnvironment defined.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Test the Environment Directly\n",
    "\n",
    "Before wiring up HTTP, test the pure game logic."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "env = WordGameEnvironment()\n",
    "obs = env.reset()\n",
    "print(f\"Word: {obs.masked_word} ({len(obs.masked_word)} letters)\")\n",
    "print(f\"Message: {obs.message}\")\n",
    "print(f\"Attempts: {obs.attempts_remaining}\")\n",
    "print()\n",
    "\n",
    "# Play with common letters\n",
    "for letter in [\"e\", \"a\", \"t\", \"n\", \"o\", \"r\", \"s\", \"i\", \"l\"]:\n",
    "    if obs.done:\n",
    "        break\n",
    "    obs = env.step(WordGameAction(guess=letter))\n",
    "    print(f\"  Guess '{letter}': {obs.masked_word}  ({obs.message})\")\n",
    "\n",
    "print(f\"\\nFinal: reward={obs.reward}, done={obs.done}\")\n",
    "print(f\"State: episode={env.state.episode_id[:8]}..., steps={env.state.step_count}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Write Policies\n",
    "\n",
    "Let's write two policies and compare them."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import string\n",
    "\n",
    "class RandomLetterPolicy:\n",
    "    \"\"\"Guess random unused letters.\"\"\"\n",
    "    name = \"Random\"\n",
    "\n",
    "    def select_action(self, obs: WordGameObservation) -> WordGameAction:\n",
    "        available = [c for c in string.ascii_lowercase if c not in obs.guessed_letters]\n",
    "        return WordGameAction(guess=random.choice(available))\n",
    "\n",
    "\n",
    "class FrequencyPolicy:\n",
    "    \"\"\"Guess by English letter frequency.\"\"\"\n",
    "    name = \"Frequency\"\n",
    "    FREQ_ORDER = \"etaoinshrdlcumwfgypbvkjxqz\"\n",
    "\n",
    "    def select_action(self, obs: WordGameObservation) -> WordGameAction:\n",
    "        for letter in self.FREQ_ORDER:\n",
    "            if letter not in obs.guessed_letters:\n",
    "                return WordGameAction(guess=letter)\n",
    "        return WordGameAction(guess=\"a\")  # fallback\n",
    "\n",
    "\n",
    "def evaluate(env, policy, episodes=100):\n",
    "    wins = 0\n",
    "    total_steps = 0\n",
    "    for _ in range(episodes):\n",
    "        obs = env.reset()\n",
    "        while not obs.done:\n",
    "            action = policy.select_action(obs)\n",
    "            obs = env.step(action)\n",
    "        if obs.reward and obs.reward > 0:\n",
    "            wins += 1\n",
    "        total_steps += env.state.step_count\n",
    "    return wins / episodes, total_steps / episodes\n",
    "\n",
    "\n",
    "env = WordGameEnvironment()\n",
    "\n",
    "for policy in [RandomLetterPolicy(), FrequencyPolicy()]:\n",
    "    win_rate, avg_steps = evaluate(env, policy)\n",
    "    print(f\"{policy.name:15s} — Win rate: {win_rate*100:.1f}%, Avg steps: {avg_steps:.1f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "Frequency should significantly outperform random. With technical vocabulary and individual letter guessing, both win rates are modest — but Frequency is typically 5–10× better than Random. Increase `max_attempts` in `WordGameEnvironment` (e.g. to 15) to see higher absolute win rates."
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Wire Up FastAPI\n",
    "\n",
    "In a real deployment, you'd create `server/app.py` with:\n",
    "\n",
    "```python\n",
    "from openenv.core.env_server import create_fastapi_app\n",
    "from environment import WordGameEnvironment\n",
    "\n",
    "app = create_fastapi_app(WordGameEnvironment)\n",
    "```\n",
    "\n",
    "That single call creates all endpoints: `/ws`, `/reset`, `/step`, `/state`, `/health`, `/web`, `/docs`.\n",
    "\n",
    "Let's simulate the server locally to demonstrate the full stack."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Write the environment files to disk for deployment\n",
    "import os\n",
    "\n",
    "os.makedirs('word_game/server', exist_ok=True)\n",
    "\n",
    "# models.py — uses Pydantic (Action, Observation, State are Pydantic BaseModel subclasses)\n",
    "models_code = '''\n",
    "from typing import List, Optional\n",
    "from openenv.core.env_server import Action, Observation, State\n",
    "\n",
    "\n",
    "class WordGameAction(Action):\n",
    "    \"\"\"Player guesses a single letter.\"\"\"\n",
    "    guess: str\n",
    "\n",
    "\n",
    "class WordGameObservation(Observation):\n",
    "    \"\"\"What the player sees after each guess.\n",
    "\n",
    "    Note: done and reward are inherited from Observation.\n",
    "    \"\"\"\n",
    "    masked_word: str            # e.g. \"p_th_n\"\n",
    "    guessed_letters: List[str]  # All letters tried\n",
    "    attempts_remaining: int\n",
    "    message: str                # Feedback text\n",
    "\n",
    "\n",
    "class WordGameState(State):\n",
    "    \"\"\"Episode metadata.\n",
    "\n",
    "    Note: episode_id and step_count are inherited from State.\n",
    "    \"\"\"\n",
    "    target_word: str = \"\"\n",
    "    max_attempts: int = 6\n",
    "'''\n",
    "\n",
    "with open('word_game/models.py', 'w') as f:\n",
    "    f.write(models_code)\n",
    "\n",
    "# client.py — uses EnvClient (WebSocket-based)\n",
    "client_code = '''\n",
    "from openenv.core.env_client import EnvClient\n",
    "from openenv.core.client_types import StepResult\n",
    "from .models import WordGameAction, WordGameObservation, WordGameState\n",
    "\n",
    "\n",
    "class WordGameEnv(EnvClient[WordGameAction, WordGameObservation, WordGameState]):\n",
    "    def _step_payload(self, action: WordGameAction) -> dict:\n",
    "        return {\"guess\": action.guess}\n",
    "\n",
    "    def _parse_result(self, payload: dict) -> StepResult:\n",
    "        obs_data = payload.get(\"observation\", {})\n",
    "        return StepResult(\n",
    "            observation=WordGameObservation(\n",
    "                done=payload.get(\"done\", False),\n",
    "                reward=payload.get(\"reward\"),\n",
    "                masked_word=obs_data.get(\"masked_word\", \"\"),\n",
    "                guessed_letters=obs_data.get(\"guessed_letters\", []),\n",
    "                attempts_remaining=obs_data.get(\"attempts_remaining\", 0),\n",
    "                message=obs_data.get(\"message\", \"\"),\n",
    "            ),\n",
    "            reward=payload.get(\"reward\"),\n",
    "            done=payload.get(\"done\", False),\n",
    "        )\n",
    "\n",
    "    def _parse_state(self, payload: dict) -> WordGameState:\n",
    "        return WordGameState(\n",
    "            episode_id=payload.get(\"episode_id\"),\n",
    "            step_count=payload.get(\"step_count\", 0),\n",
    "            target_word=payload.get(\"target_word\", \"\"),\n",
    "            max_attempts=payload.get(\"max_attempts\", 6),\n",
    "        )\n",
    "'''\n",
    "\n",
    "with open('word_game/client.py', 'w') as f:\n",
    "    f.write(client_code)\n",
    "\n",
    "# server/app.py\n",
    "app_code = '''\n",
    "from openenv.core.env_server import create_fastapi_app\n",
    "from ..models import WordGameAction, WordGameObservation\n",
    "from .environment import WordGameEnvironment\n",
    "\n",
    "app = create_fastapi_app(WordGameEnvironment, WordGameAction, WordGameObservation)\n",
    "'''\n",
    "\n",
    "with open('word_game/server/app.py', 'w') as f:\n",
    "    f.write(app_code)\n",
    "\n",
    "print('Created word_game/models.py  (Pydantic models)')\n",
    "print('Created word_game/client.py  (EnvClient subclass)')\n",
    "print('Created word_game/server/app.py')\n",
    "print()\n",
    "print('Next steps:')\n",
    "print('  1. Add server/environment.py with WordGameEnvironment class')\n",
    "print('  2. Test locally: uvicorn word_game.server.app:app --reload')\n",
    "print('  3. Deploy: openenv push --repo-id username/word-game')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. The Client\n\nThe client translates between your typed models and JSON over the wire. Three methods:\n\n```python\nclass WordGameEnv(EnvClient[WordGameAction, WordGameObservation, WordGameState]):\n    def _step_payload(self, action):\n        return {\"guess\": action.guess}\n\n    def _parse_result(self, payload):\n        return StepResult(\n            observation=WordGameObservation(**payload),\n            reward=payload.get(\"reward\", 0),\n            done=payload[\"done\"],\n        )\n\n    def _parse_state(self, payload):\n        return WordGameState(**payload)\n```\n\nUsers of your environment would then write:\n\n```python\nfrom word_game import WordGameEnv, WordGameAction\n\nwith WordGameEnv(base_url=\"https://username-word-game.hf.space\").sync() as env:\n    result = env.reset()\n    result = env.step(WordGameAction(guess=\"e\"))\n    print(result.observation.masked_word)\n```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Scaffold with `openenv init`\n",
    "\n",
    "Instead of writing everything by hand, use the CLI:\n",
    "\n",
    "```bash\n",
    "openenv init word_game\n",
    "cd word_game\n",
    "# Edit models.py, server/environment.py, client.py\n",
    "uv run server           # Test locally\n",
    "openenv push             # Deploy to HF Spaces\n",
    "```\n",
    "\n",
    "This creates the full directory structure. You fill in your types and game logic."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Summary\n",
    "\n",
    "You built a complete OpenEnv environment:\n",
    "\n",
    "| File | What it does | Lines of code |\n",
    "|------|-------------|---------------|\n",
    "| `models.py` | Action, Observation, State types | ~30 |\n",
    "| `server/environment.py` | Game logic (reset, step, state) | ~60 |\n",
    "| `client.py` | HTTP client (3 parsing methods) | ~25 |\n",
    "| `server/app.py` | FastAPI wiring | ~3 |\n",
    "\n",
    "The pattern is always the same: **types → server logic → client → container**.\n",
    "\n",
    "**Next:** [Module 5](../module-5/README.md) — Training a model to play games with GRPO."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

## openenv-course/module-4/README.md

# Module 4: Building Your Own Environment

## The 3-Component Pattern

Every OpenEnv environment has the same structure:

```
my_env/
├── models.py              ← Types: Action, Observation, State
├── client.py              ← HTTP/WebSocket client (what users import)
├── server/
│   ├── environment.py     ← Game logic (reset, step, state)
│   ├── app.py             ← FastAPI server
│   └── Dockerfile         ← Container definition
├── openenv.yaml           ← Manifest
└── pyproject.toml         ← Package metadata
```

You'll build all of these for a word-guessing game. ~100 lines of meaningful code.

## Step 1: Define Your Types (`models.py`)

Start with the data contracts. What does an action look like? What does an observation contain?

```python
from typing import List, Optional
from openenv.core.env_server import Action, Observation, State

# Action, Observation, State are Pydantic BaseModel subclasses —
# no @dataclass decorator needed; define fields directly as class attributes.

class WordGameAction(Action):
    guess: str  # The player's guessed letter

class WordGameObservation(Observation):
    # done: bool and reward: Optional[float] are already in Observation base
    masked_word: str           # e.g., "h_ll_"
    guessed_letters: List[str] # Letters tried so far
    attempts_remaining: int
    message: str               # Feedback message

class WordGameState(State):
    # episode_id: Optional[str] and step_count: int are already in State base
    target_word: str = ""
    max_attempts: int = 10
```

These Pydantic models do three things:
1. **Document the API** — anyone reading `models.py` knows the interface
2. **Enable IDE autocomplete** — `obs.masked_word` instead of `obs["masked_word"]`
3. **Catch bugs at type-check time** — misspell a field name and your linter tells you

## Step 2: Implement the Environment (`server/environment.py`)

The environment implements `reset()`, `step()`, and `state`. This is where your game logic lives.

```python
import random
import uuid
from openenv.core.env_server import Environment
from .models import WordGameAction, WordGameObservation, WordGameState

WORDS = ["python", "neural", "tensor", "matrix", "vector",
         "kernel", "lambda", "signal", "binary", "cipher"]

class WordGameEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True  # Allow multiple simultaneous clients

    MAX_ATTEMPTS = 10

    def __init__(self):
        self._state = WordGameState()
        self._target = ""
        self._guessed = set()
        self._remaining = self.MAX_ATTEMPTS

    def reset(self, seed=None, episode_id=None, **kwargs) -> WordGameObservation:
        self._target = random.choice(WORDS)
        self._guessed = set()
        self._remaining = self.MAX_ATTEMPTS
        self._state = WordGameState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            target_word=self._target,
            max_attempts=self.MAX_ATTEMPTS,
        )
        return WordGameObservation(
            done=False,
            reward=None,
            masked_word=self._mask(),
            guessed_letters=[],
            attempts_remaining=self._remaining,
            message=f"Guess letters in a {len(self._target)}-letter word!",
        )

    def step(self, action: WordGameAction, timeout_s=None, **kwargs) -> WordGameObservation:
        letter = action.guess.lower().strip()
        self._state.step_count += 1
        self._guessed.add(letter)

        if letter in self._target:
            message = f"'{letter}' is in the word!"
        else:
            self._remaining -= 1
            message = f"'{letter}' is not in the word."

        # Check win/lose
        masked = self._mask()
        won = "_" not in masked
        lost = self._remaining <= 0
        done = won or lost

        if won:
            reward = 1.0
            message = f"You got it! The word was '{self._target}'."
        elif lost:
            reward = 0.0
            message = f"Out of attempts. The word was '{self._target}'."
        else:
            reward = 0.0

        return WordGameObservation(
            done=done,
            reward=reward,
            masked_word=masked,
            guessed_letters=sorted(self._guessed),
            attempts_remaining=self._remaining,
            message=message,
        )

    @property
    def state(self) -> WordGameState:
        return self._state

    def _mask(self) -> str:
        return "".join(c if c in self._guessed else "_" for c in self._target)
```

## Step 3: Create the Client (`client.py`)

The client translates between your typed models and the WebSocket wire format. Three abstract methods to implement:

```python
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import WordGameAction, WordGameObservation, WordGameState

class WordGameEnv(EnvClient[WordGameAction, WordGameObservation, WordGameState]):
    def _step_payload(self, action: WordGameAction) -> dict:
        return {"guess": action.guess}

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=WordGameObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                masked_word=obs_data.get("masked_word", ""),
                guessed_letters=obs_data.get("guessed_letters", []),
                attempts_remaining=obs_data.get("attempts_remaining", 0),
                message=obs_data.get("message", ""),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> WordGameState:
        return WordGameState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            target_word=payload.get("target_word", ""),
            max_attempts=payload.get("max_attempts", 6),
        )
```

That's it. The `EnvClient` base class handles all WebSocket communication.

## Step 4: Wire Up FastAPI (`server/app.py`)

One line of meaningful code:

```python
from openenv.core.env_server import create_fastapi_app
from environment import WordGameEnvironment

app = create_fastapi_app(WordGameEnvironment)
```

`create_fastapi_app()` creates all the endpoints: `/ws`, `/reset`, `/step`, `/state`, `/health`, `/web`, `/docs`.

## Step 5: Dockerize (`server/Dockerfile`)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## The Fast Path: `openenv init`

Don't want to write all this by hand? Scaffold it:

```bash
openenv init word_game
cd word_game
```

This creates the full directory structure with placeholder code. You just fill in:
1. Your types in `models.py`
2. Your game logic in `server/environment.py`
3. Your client parsing in `client.py`

Then test and deploy:
```bash
uv run server                    # Test locally
openenv push --repo-id user/word-game  # Deploy
```

## What's Next

In the [notebook](notebook.ipynb), you'll scaffold a word game with `openenv init`, implement the game logic, test it locally, and deploy it.

**Key takeaway:** The pattern is always the same — types, server logic, client, container. ~100 lines of meaningful code for a custom environment.


## openenv-course/module-5/notebook.ipynb

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Module 5: Train a Wordle Agent with GRPO\n",
    "\n",
    "Fine-tune Qwen3-1.7B to play Wordle using GRPO (Group Relative Policy Optimization) via TRL and OpenEnv.\n",
    "\n",
    "**Time:** ~90 min (training) · **Difficulty:** Advanced · **GPU:** A100 required (Colab Pro or similar)\n",
    "\n",
    "Based on the [TRL OpenEnv Wordle example](https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "# Requires GPU (A100 recommended). Run on Colab Pro or similar.\n!pip install -Uq \"trl>=0.17.0\" openenv-core transformers datasets accelerate vllm trackio\n!git clone --depth=1 -q https://github.com/meta-pytorch/OpenEnv.git 2>/dev/null || true\n\nimport sys, os\nrepo = os.path.abspath('OpenEnv')\nfor p in [repo, os.path.join(repo, 'src')]:\n    if p not in sys.path:\n        sys.path.insert(0, p)\nprint(\"Setup complete!\")"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Log in to Hugging Face (required for model access and pushing results)\n",
    "from huggingface_hub import notebook_login\n",
    "notebook_login()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Initialize the Environment\n",
    "\n",
    "Connect to the TextArena Wordle environment hosted on HF Spaces.\n",
    "\n",
    "> **For production use:** Duplicate the Space to your own account to avoid concurrency limits."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from envs.textarena_env import TextArenaEnv\n",
    "\n",
    "textarena_url = 'https://burtenshaw-textarena.hf.space'  # Duplicate this Space for production use!\n",
    "\n",
    "# Verify connection\n",
    "with TextArenaEnv(base_url=textarena_url).sync() as _check:\n",
    "    result = _check.reset()\n",
    "    print(f'Connected to: {textarena_url}')\n",
    "    print(f'Prompt preview: {str(result.observation.prompt)[:100]}...')\n",
    "\n",
    "# Create a persistent sync client for training.\n",
    "# A single WebSocket connection is reused across all rollouts instead of\n",
    "# opening/closing one per episode, which matters at training throughput.\n",
    "env = TextArenaEnv(base_url=textarena_url)\n",
    "sync_env = env.sync()\n",
    "sync_env.connect()\n",
    "print('Persistent training connection established.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Init Model and Tokenizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import AutoTokenizer\n",
    "\n",
    "model_name = \"Qwen/Qwen3-1.7B\"\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
    "tokenizer.pad_token = tokenizer.eos_token\n",
    "print(f\"Model: {model_name}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Define the System Prompt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "system_prompt = \"\"\"\n",
    "You are an expert Wordle solver with deep knowledge of English vocabulary, letter frequency patterns, and optimal guessing strategies.\n",
    "\n",
    "## GAME RULES\n",
    "\n",
    "1. The target is a 5-letter English word\n",
    "2. You have 6 attempts to guess the correct word\n",
    "3. After each guess, you receive color-coded feedback:\n",
    "   - GREEN: Letter is correct and in the correct position\n",
    "   - YELLOW: Letter is in the word but in the wrong position\n",
    "   - GRAY: Letter is not in the word at all\n",
    "4. All guesses must be valid 5-letter English words\n",
    "5. You cannot reuse a word you've already guessed\n",
    "\n",
    "## RESPONSE FORMAT\n",
    "\n",
    "Only respond with your next guess in square brackets, e.g., [crane].\n",
    "\n",
    "## STRATEGIC APPROACH\n",
    "\n",
    "Do not repeat the same guess twice.\n",
    "\n",
    "### Opening Strategy\n",
    "- Start with words rich in common vowels (A, E, I, O, U) and consonants (R, S, T, L, N)\n",
    "- Optimal starters: CRANE, SLATE, STARE, AROSE, IRATE\n",
    "\n",
    "### Mid-Game Strategy\n",
    "- Use confirmed GREEN letters in their correct positions\n",
    "- Place YELLOW letters in different positions than where they appeared\n",
    "- Eliminate GRAY letters from consideration\n",
    "\n",
    "## YOUR GOAL\n",
    "\n",
    "Solve the Wordle in as few guesses as possible.\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Helper Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def make_user_prompt(prompt_text, messages):\n",
    "    \"\"\"Build a structured prompt from game state and message history.\"\"\"\n",
    "    history = format_history(messages)\n",
    "    prompt_section = prompt_text.strip() if prompt_text.strip() else \"Wordle-v0\"\n",
    "    history_section = history if history else \"[PROMPT] Awaiting first feedback.\"\n",
    "    return (\n",
    "        f\"Game prompt:\\n{prompt_section}\\n\\n\"\n",
    "        f\"Conversation so far:\\n{history_section}\\n\\n\"\n",
    "        \"Reply with your next guess enclosed in square brackets.\"\n",
    "    )\n",
    "\n",
    "\n",
    "def format_history(messages):\n",
    "    \"\"\"Format message history with category tags.\"\"\"\n",
    "    lines = []\n",
    "    for message in messages:\n",
    "        tag = message.category or \"MESSAGE\"\n",
    "        content = message.content.strip()\n",
    "        if content:\n",
    "            lines.append(f\"[{tag}] {content}\")\n",
    "    return \"\\n\".join(lines)\n",
    "\n",
    "\n",
    "def scale_repetition_score(previous_occurrences, max_occurrences):\n",
    "    \"\"\"Scale repetition penalty: 1.0 = novel guess, 0.0 = fully repeated.\"\"\"\n",
    "    if max_occurrences == 0:\n",
    "        return 0.0\n",
    "    return (max_occurrences - previous_occurrences) / max_occurrences\n",
    "\n",
    "\n",
    "print(\"Helper functions defined.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Define the Rollout Function\n",
    "\n",
    "The rollout function plays one full Wordle game per prompt. It's called by `GRPOTrainer` during training."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from collections import defaultdict\n",
    "from envs.textarena_env.models import TextArenaAction\n",
    "from envs.textarena_env.rewards import extract_feedback_counts, extract_guess, extract_wordle_feedback\n",
    "from trl.experimental.openenv import generate_rollout_completions\n",
    "\n",
    "\n",
    "def rollout_once(trainer, sync_env, tokenizer, dataset_prompt, system_prompt, max_turns):\n",
    "    \"\"\"Execute one full Wordle episode using an already-connected sync client.\"\"\"\n",
    "    result = sync_env.reset()\n",
    "    observation = result.observation\n",
    "\n",
    "    prompt_ids = []\n",
    "    completion_ids = []\n",
    "    logprobs = []\n",
    "    green_scores = []\n",
    "    yellow_scores = []\n",
    "    repetition_scores = []\n",
    "    correct_scores = []\n",
    "    guess_counts = defaultdict(int)\n",
    "\n",
    "    for _turn in range(max_turns):\n",
    "        if result.done:\n",
    "            break\n",
    "\n",
    "        base_prompt = observation.prompt or dataset_prompt\n",
    "        user_prompt = make_user_prompt(base_prompt, observation.messages)\n",
    "        messages = [\n",
    "            {'role': 'system', 'content': system_prompt},\n",
    "            {'role': 'user', 'content': user_prompt},\n",
    "        ]\n",
    "        prompt_text = tokenizer.apply_chat_template(\n",
    "            messages,\n",
    "            add_generation_prompt=True,\n",
    "            tokenize=False,\n",
    "            enable_thinking=False,\n",
    "        )\n",
    "\n",
    "        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]\n",
    "        prompt_ids.extend(rollout_outputs['prompt_ids'])\n",
    "        completion_ids.extend(rollout_outputs['completion_ids'])\n",
    "        logprobs.extend(rollout_outputs['logprobs'])\n",
    "        completion_text = rollout_outputs.get('text') or tokenizer.decode(\n",
    "            rollout_outputs['completion_ids'], skip_special_tokens=True\n",
    "        )\n",
    "\n",
    "        guess = extract_guess(completion_text)\n",
    "        result = sync_env.step(TextArenaAction(message=guess))\n",
    "        observation = result.observation\n",
    "        correct_score = float(result.reward or 0.0)\n",
    "        feedback = extract_wordle_feedback(observation)\n",
    "\n",
    "        previous_occurrences = guess_counts[guess]\n",
    "        repetition_score = max(0.0, 1.0 - previous_occurrences)\n",
    "        guess_counts[guess] += 1\n",
    "\n",
    "        if not feedback:\n",
    "            green_score, yellow_score = 0.0, 0.0\n",
    "        else:\n",
    "            green_count, yellow_count = extract_feedback_counts(feedback)\n",
    "            green_score = green_count / 5.0\n",
    "            yellow_score = yellow_count / 5.0\n",
    "\n",
    "        repetition_scores.append(repetition_score)\n",
    "        green_scores.append(green_score)\n",
    "        yellow_scores.append(yellow_score)\n",
    "        correct_scores.append(correct_score)\n",
    "\n",
    "    return {\n",
    "        'prompt_ids': prompt_ids,\n",
    "        'completion_ids': completion_ids,\n",
    "        'logprobs': logprobs,\n",
    "        'correct_reward': correct_scores[-1] if correct_scores else 0.0,\n",
    "        'green_reward': green_scores[-1] if green_scores else 0.0,\n",
    "        'yellow_reward': yellow_scores[-1] if yellow_scores else 0.0,\n",
    "        'repetition_reward': repetition_scores[-1] if repetition_scores else 0.0,\n",
    "    }\n",
    "\n",
    "\n",
    "def rollout_func(prompts, trainer=None):\n",
    "    \"\"\"Rollout function called by GRPOTrainer. Uses the module-level sync_env.\"\"\"\n",
    "    episode_prompt_ids = []\n",
    "    episode_completion_ids = []\n",
    "    episode_logprobs = []\n",
    "    correctness_rewards = []\n",
    "    green_rewards = []\n",
    "    yellow_rewards = []\n",
    "    repetition_rewards = []\n",
    "\n",
    "    for prompt_text in prompts:\n",
    "        episode = rollout_once(\n",
    "            trainer=trainer,\n",
    "            sync_env=sync_env,     # Persistent connection — no reconnect per episode\n",
    "            tokenizer=tokenizer,\n",
    "            dataset_prompt=prompt_text,\n",
    "            system_prompt=system_prompt,\n",
    "            max_turns=6,\n",
    "        )\n",
    "        episode_prompt_ids.append(episode['prompt_ids'])\n",
    "        episode_completion_ids.append(episode['completion_ids'])\n",
    "        episode_logprobs.append(episode['logprobs'])\n",
    "        correctness_rewards.append(episode['correct_reward'])\n",
    "        green_rewards.append(episode['green_reward'])\n",
    "        yellow_rewards.append(episode['yellow_reward'])\n",
    "        repetition_rewards.append(episode['repetition_reward'])\n",
    "\n",
    "    return {\n",
    "        'prompt_ids': episode_prompt_ids,\n",
    "        'completion_ids': episode_completion_ids,\n",
    "        'logprobs': episode_logprobs,\n",
    "        'correct_reward': correctness_rewards,\n",
    "        'green_reward': green_rewards,\n",
    "        'yellow_reward': yellow_rewards,\n",
    "        'repetition_reward': repetition_rewards,\n",
    "    }\n",
    "\n",
    "\n",
    "print('Rollout functions defined.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Define Reward Functions\n",
    "\n",
    "Four reward signals for richer gradient information."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def reward_correct(completions, **kwargs):\n",
    "    rewards = kwargs.get(\"correct_reward\")\n",
    "    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)\n",
    "\n",
    "def reward_greens(completions, **kwargs):\n",
    "    rewards = kwargs.get(\"green_reward\")\n",
    "    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)\n",
    "\n",
    "def reward_yellows(completions, **kwargs):\n",
    "    rewards = kwargs.get(\"yellow_reward\")\n",
    "    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)\n",
    "\n",
    "def reward_repetition(completions, **kwargs):\n",
    "    rewards = kwargs.get(\"repetition_reward\")\n",
    "    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)\n",
    "\n",
    "print(\"Reward functions: correct, greens, yellows, repetition\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Create Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from datasets import Dataset\n",
    "\n",
    "dataset_size = 1000\n",
    "dataset = Dataset.from_dict({\"prompt\": [\"Play Wordle like an expert.\"] * dataset_size})\n",
    "print(f\"Dataset: {len(dataset)} prompts\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Configure GRPO Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from trl import GRPOConfig\n",
    "\n",
    "output_dir = \"wordle-grpo-Qwen3-1.7B\"\n",
    "\n",
    "grpo_config = GRPOConfig(\n",
    "    num_train_epochs=1,\n",
    "    learning_rate=5e-6,\n",
    "    gradient_accumulation_steps=64,\n",
    "    per_device_train_batch_size=1,\n",
    "    warmup_steps=20,\n",
    "    num_generations=2,\n",
    "    max_completion_length=8,\n",
    "    max_prompt_length=1400,\n",
    "    use_vllm=True,\n",
    "    vllm_mode=\"colocate\",\n",
    "    vllm_gpu_memory_utilization=0.1,\n",
    "    output_dir=output_dir,\n",
    "    report_to=\"trackio\",\n",
    "    trackio_space_id=output_dir,\n",
    "    logging_steps=1,\n",
    "    save_steps=10,\n",
    "    gradient_checkpointing=True,\n",
    "    gradient_checkpointing_kwargs={\"use_reentrant\": False},\n",
    "    push_to_hub=True,\n",
    ")\n",
    "\n",
    "print(f\"Output: {output_dir}\")\n",
    "print(f\"vLLM mode: colocate (generation + training on same GPU)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Create Trainer and Train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from trl import GRPOTrainer\n",
    "\n",
    "trainer = GRPOTrainer(\n",
    "    model=model_name,\n",
    "    processing_class=tokenizer,\n",
    "    reward_funcs=[\n",
    "        reward_correct,\n",
    "        reward_greens,\n",
    "        reward_yellows,\n",
    "        reward_repetition,\n",
    "    ],\n",
    "    train_dataset=dataset,\n",
    "    args=grpo_config,\n",
    "    rollout_func=rollout_func,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check GPU before training\n",
    "import torch\n",
    "gpu_stats = torch.cuda.get_device_properties(0)\n",
    "start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)\n",
    "max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)\n",
    "print(f\"GPU: {gpu_stats.name} — {max_memory} GB total, {start_gpu_memory} GB reserved\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train (~90 minutes on A100)\n",
    "trainer_stats = trainer.train()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Memory stats after training\n",
    "used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)\n",
    "used_for_training = round(used_memory - start_gpu_memory, 3)\n",
    "\n",
    "print(f\"Training time: {round(trainer_stats.metrics['train_runtime']/60, 1)} minutes\")\n",
    "print(f\"Peak memory: {used_memory} GB ({round(used_memory/max_memory*100, 1)}% of {max_memory} GB)\")\n",
    "print(f\"Memory for training: {used_for_training} GB\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Save and Push"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Close the persistent environment connection before saving\n",
    "sync_env.close()\n",
    "\n",
    "trainer.save_model(output_dir)\n",
    "trainer.push_to_hub()\n",
    "print(f'Model saved to {output_dir} and pushed to Hub.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 11. Evaluate: Play Wordle with the Trained Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import AutoModelForCausalLM\n",
    "from envs.textarena_env.models import TextArenaAction\n",
    "from envs.textarena_env.rewards import extract_guess\n",
    "\n",
    "# Load the fine-tuned model (replace with your HF repo id if you pushed)\n",
    "fine_tuned_model = AutoModelForCausalLM.from_pretrained(\n",
    "    output_dir, torch_dtype='auto', device_map='auto'\n",
    ")\n",
    "\n",
    "\n",
    "def play_wordle(sync_env, model, tokenizer, max_turns=6):\n",
    "    \"\"\"Play one Wordle game and print each turn.\"\"\"\n",
    "    result = sync_env.reset()\n",
    "    observation = result.observation\n",
    "    print(f'Prompt: {observation.prompt[:100]}...')\n",
    "\n",
    "    for turn in range(max_turns):\n",
    "        if result.done:\n",
    "            break\n",
    "\n",
    "        user_prompt = make_user_prompt(observation.prompt, observation.messages)\n",
    "        messages = [\n",
    "            {'role': 'system', 'content': system_prompt},\n",
    "            {'role': 'user', 'content': user_prompt},\n",
    "        ]\n",
    "        prompt_text = tokenizer.apply_chat_template(\n",
    "            messages, add_generation_prompt=True,\n",
    "            tokenize=False, enable_thinking=False,\n",
    "        )\n",
    "\n",
    "        model_inputs = tokenizer([prompt_text], return_tensors='pt').to(model.device)\n",
    "        generated_ids = model.generate(**model_inputs, max_new_tokens=512)\n",
    "        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]\n",
    "        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)\n",
    "        guess = extract_guess(generated_text)\n",
    "\n",
    "        print(f'\\nTurn {turn + 1}: {guess}')\n",
    "        result = sync_env.step(TextArenaAction(message=guess))\n",
    "        observation = result.observation\n",
    "        for msg in observation.messages:\n",
    "            print(f'  [{msg.category}] {msg.content}')\n",
    "\n",
    "    print(f'\\nResult: reward={result.reward}, done={result.done}')\n",
    "\n",
    "\n",
    "# Evaluation uses a fresh per-game context (not the training connection)\n",
    "eval_env = TextArenaEnv(base_url=textarena_url)\n",
    "with eval_env.sync() as eval_sync:\n",
    "    play_wordle(eval_sync, fine_tuned_model, tokenizer)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Summary\n",
    "\n",
    "What you did:\n",
    "1. Connected to the TextArena Wordle environment via OpenEnv\n",
    "2. Defined a system prompt, rollout function, and 4 reward signals\n",
    "3. Trained Qwen3-1.7B with GRPO for ~90 minutes on an A100\n",
    "4. Evaluated the trained model on live Wordle games\n",
    "\n",
    "The key insight: **OpenEnv makes the environment a plug-in.** Swap Wordle for any other OpenEnv environment — your Module 4 word game, a coding environment, a math problem — and the training pipeline stays the same.\n",
    "\n",
    "### What's next\n",
    "\n",
    "- **Improve the model:** Longer training, larger models, better reward shaping\n",
    "- **Build your own environment:** Use Module 4's pattern, plug it into this pipeline\n",
    "- **Scale up:** See the [Scaling appendix](../README.md#bonus-scaling-openenv) for multi-container deployment\n",
    "- **Explore the Hub:** Browse [openenv environments](https://huggingface.co/collections/openenv/environment-hub) for inspiration"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  },
  "accelerator": "GPU",
  "gpuClass": "premium"
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

## openenv-course/module-5/README.md

# Module 5: Training with OpenEnv + TRL

## What is GRPO?

**Group Relative Policy Optimization** is a reinforcement learning algorithm for fine-tuning LLMs. The intuition:

1. Generate a group of completions for the same prompt
2. Score each completion with reward functions
3. Use the relative ranking within the group to update the policy

No value model needed (unlike PPO). The group itself provides the baseline.

GRPO works well for tasks where you can define reward functions — games, code generation, reasoning, structured output.

## The TRL + OpenEnv Integration

[TRL (Transformers Reinforcement Learning)](https://github.com/huggingface/trl) provides `GRPOTrainer` with native OpenEnv support. The key abstraction is the **rollout function** — it defines how the model interacts with the environment during training.

The loop:
1. `GRPOTrainer` calls your rollout function with prompts
2. Your function generates completions using the model
3. Each completion is sent as an action to the environment
4. The environment returns observations + rewards
5. TRL uses the rewards to update the model

```python
trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[reward_correct, reward_greens, reward_yellows],
    rollout_func=rollout_func,   # Your environment interaction
    train_dataset=dataset,
    args=grpo_config,
)
trainer.train()
```

## The Wordle Training Pipeline

We'll train Qwen3-1.7B to play Wordle using the TextArena environment.

### Environment Setup

```python
from envs.textarena_env import TextArenaEnv

env = TextArenaEnv(base_url="https://burtenshaw-textarena.hf.space")
```

The TextArena Wordle environment:
- Accepts guesses as `[word]` (5-letter words in brackets)
- Returns feedback: G (green), Y (yellow), X (gray) for each letter
- 6 attempts per game
- Reward: 1.0 for correct guess, 0.0 otherwise

### System Prompt

The system prompt guides the model's strategy:

```python
system_prompt = """
You are an expert Wordle solver.

RULES:
- Guess a 5-letter English word
- Feedback: GREEN (correct position), YELLOW (wrong position), GRAY (not in word)
- 6 attempts maximum

RESPONSE FORMAT:
Only respond with your guess in square brackets, e.g., [crane]

STRATEGY:
- Start with vowel-rich words: CRANE, SLATE, STARE
- Use GREEN letters in their positions
- Move YELLOW letters to new positions
- Eliminate GRAY letters
"""
```

### Reward Functions

Multiple reward signals give the model richer gradient information:

| Reward | What it measures | Range |
|--------|-----------------|-------|
| `reward_correct` | Did the model solve it? | 0.0 or 1.0 |
| `reward_greens` | How many green letters? | 0.0 to 1.0 |
| `reward_yellows` | How many yellow letters? | 0.0 to 1.0 |
| `reward_repetition` | Penalize repeated guesses | 0.0 to 1.0 |

Greens and yellows provide shaping signal even when the model doesn't win. Repetition penalty discourages the model from guessing the same word twice.

### The Rollout Function

The rollout function plays one full Wordle game:

```python
def rollout_once(trainer, env, tokenizer, prompt, system_prompt, max_turns):
    result = env.reset()
    observation = result.observation

    for turn in range(max_turns):
        if result.done:
            break

        # Build prompt from game state
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": format_game_state(observation)},
        ]

        # Generate with the model
        rollout = generate_rollout_completions(trainer, [messages])

        # Parse guess and send to environment
        guess = extract_guess(rollout["text"])
        result = env.step(TextArenaAction(message=guess))
        observation = result.observation

    return {
        "prompt_ids": ..., "completion_ids": ..., "logprobs": ...,
        "correct_reward": ..., "green_reward": ...,
    }
```

### GRPO Configuration

```python
grpo_config = GRPOConfig(
    num_train_epochs=1,
    learning_rate=5e-6,
    gradient_accumulation_steps=64,
    per_device_train_batch_size=1,
    num_generations=2,
    max_completion_length=8,
    max_prompt_length=1400,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.1,
    gradient_checkpointing=True,
    report_to="trackio",
)
```

Key settings:
- **vLLM colocate mode** — generation and training share one GPU
- **gradient_accumulation_steps=64** — effective batch size without OOM
- **max_completion_length=8** — Wordle guesses are short

### Hardware

- **GPU:** A100 40GB (Colab Pro or similar)
- **Training time:** ~90 minutes
- **Peak memory:** ~37GB

## What the Model Learns

After training:
- Opens with strong words (CRANE, SLATE)
- Uses feedback to narrow down candidates
- Places confirmed letters in correct positions
- Still struggles with repeated guesses (common RL challenge)

This is a starting point. Improvements:
- Longer training runs
- Stronger repetition penalties
- Larger models (Qwen3-8B, etc.)
- Custom environments (swap Wordle for anything)

## The Key Insight

OpenEnv makes the environment a plug-in. The training pipeline stays the same — swap Wordle for your Module 4 word game, a coding environment, a math problem, or anything else. The `rollout_func` interface is the same.

## What's Next

In the [notebook](notebook.ipynb), you'll run the full Wordle GRPO training pipeline on an A100.

**Key takeaway:** OpenEnv + TRL gives you a standard way to train LLMs with environment feedback. Build the environment (Modules 1-4), plug it into GRPO, train.


## openenv-course/scripts/validate_notebooks.py

#!/usr/bin/env python3
"""
Validate code cells in Jupyter notebooks.

Usage:
    python scripts/validate_notebooks.py

Behaviour:
    1. Find all *.ipynb files in the repo.
    2. Extract every code cell.
    3. Syntax-check every cell (compile).
    4. For cells that have no network calls / subprocess / LLM imports,
       also *execute* them in a shared namespace per notebook
       (so cells can depend on definitions from earlier cells).
    5. Report pass / skip / fail for every cell.
"""

import json
import re
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent

SKIP_EXECUTION_PATTERNS = [
    # shell / network
    r"!pip",
    r"!git",
    r"!find",
    r"!cat",
    r"subprocess",
    r"uvicorn",
    r"notebook_login",
    # live environment connections
    r"\.hf\.space",
    r"localhost:\d+",
    r"EnvClient\(",
    r"TextArenaEnv\(",
    r"OpenSpielEnv\(",
    r"EchoEnv\(",
    # LLM / GPU
    r"AutoModelForCausalLM",
    r"AutoTokenizer",
    r"GRPOTrainer",
    r"GRPOConfig",
    r"trainer\.",
    r"from_pretrained",
    r"torch\.cuda",
    r"generate_rollout_completions",
    # imports that won't be available locally
    r"from envs\.",
    r"from openenv",
    r"from trl",
    r"from transformers",
    r"from datasets",
    r"from huggingface_hub",
    # file system mutations or reads that depend on prior cloned/created files
    r'os\.makedirs',
    r'open\(.+, *["\']w["\']',
    r'open\(env_file',
    r"shutil\.",
    r"echo-env-modified",
    r"word_game/",
    # cleanup / teardown depending on skipped setup
    r"server\.terminate",
    r"server\.wait",
    r"server\.pid",
    # training utilities
    r"import trackio",
]

_SKIP_RE = re.compile("|".join(SKIP_EXECUTION_PATTERNS))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_notebooks():
    return sorted(REPO_ROOT.rglob("*.ipynb"))


def extract_code_cells(nb_path: Path):
    """Return list of (cell_index, source_str) for code cells."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if source.strip():
                cells.append((i, source))
    return cells


def should_skip_execution(code: str) -> bool:
    return bool(_SKIP_RE.search(code))


def syntax_check(code: str, label: str) -> tuple[bool, str]:
    # Strip Jupyter magic / shell commands for syntax check
    lines = []
    for line in code.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            lines.append(f"# SHELL: {line}")
        else:
            lines.append(line)
    cleaned = "\n".join(lines)
    try:
        compile(cleaned, label, "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def execute_cell(code: str, ns: dict, label: str) -> tuple[bool, str]:
    # Strip shell / magic lines
    lines = []
    for line in code.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            lines.append(f"# SHELL: {line}")
        else:
            lines.append(line)
    cleaned = "\n".join(lines)
    try:
        exec(compile(cleaned, label, "exec"), ns)  # noqa: S102
        return True, ""
    except Exception:
        return False, traceback.format_exc(limit=5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    notebooks = find_notebooks()
    total = 0
    passed = 0
    skipped = 0
    failed = 0
    failures = []

    for nb_path in notebooks:
        rel = nb_path.relative_to(REPO_ROOT)
        print(f"\n{'='*60}")
        print(f"Notebook: {rel}")
        print("=" * 60)

        cells = extract_code_cells(nb_path)
        if not cells:
            print("  (no code cells)")
            continue

        # Shared namespace so cells can reference each other
        nb_ns = {}

        for cell_idx, code in cells:
            total += 1
            label = f"cell[{cell_idx}]"

            # 1. Syntax check
            ok, err = syntax_check(code, label)
            if not ok:
                failed += 1
                failures.append((f"{rel} {label}", "SYNTAX", err, code))
                print(f"  FAIL  {label}: {err}")
                continue

            # 2. Execution (only for pure-logic cells)
            if should_skip_execution(code):
                skipped += 1
                print(f"  SKIP  {label}")
                continue

            ok, err = execute_cell(code, nb_ns, label)
            if ok:
                passed += 1
                print(f"  PASS  {label}")
            else:
                failed += 1
                failures.append((f"{rel} {label}", "RUNTIME", err, code))
                print(f"  FAIL  {label}")
                for line in err.strip().split("\n")[:4]:
                    print(f"        {line}")

    print()
    print("=" * 70)
    print(f"Results: {total} cells | {passed} PASS | {skipped} SKIP | {failed} FAIL")
    print("=" * 70)

    if failures:
        print("\nFailed cells:\n")
        for label, kind, err, code in failures:
            print(f"--- {kind}: {label} ---")
            print(err.strip())
            print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())


## openenv-course/scripts/validate_snippets.py

#!/usr/bin/env python3
"""
Validate Python code blocks extracted from Markdown files.

Usage:
    python scripts/validate_snippets.py [--fix]

Behaviour:
    1. Find all *.md files in the repo (excluding .git).
    2. Extract every ```python ... ``` fenced code block.
    3. Syntax-check every block (compile).
    4. For blocks that have no network calls / subprocess / LLM imports,
       also *execute* them in an isolated namespace.
    5. Report pass / skip / fail for every block.

Network / LLM blocks are detected heuristically and only syntax-checked —
they are marked SKIP (execution) but still must parse cleanly.
"""

import ast
import re
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SKIP_EXECUTION_PATTERNS = [
    # network / live environment calls
    r"\.hf\.space",
    r"localhost:\d+",
    r"subprocess",
    r"uvicorn",
    r"openenv push",
    r"git clone",
    # LLM / GPU
    r"AutoModelForCausalLM",
    r"GRPOTrainer",
    r"GRPOConfig",
    r"trainer\.train",
    r"notebook_login",
    r"from_pretrained",
    r"torch\.cuda",
    # package installation (side effects)
    r"!pip",
    r"pip install",
    # file system mutations we don't want
    r'os\.makedirs',
    r'open\(.+, *["\']w["\']',
    # imports that won't be available locally
    r"from envs\.",
    r"from openenv",
    r"from trl",
    r"from transformers",
    r"from datasets",
    r"import trackio",
    # pseudo-code / class skeletons with undefined names
    r"class \w+\(ABC\)",
    r"class \w+\(Environment\)",
    r"class \w+\(EnvClient\)",
    r"class \w+\(HTTPEnvClient\)",
    r"while not done:",
    r"policy\.choose",
    r"environment\.observe",
    # standalone env calls without context
    r"^env\.(reset|step|state)\(",
]

_SKIP_RE = re.compile("|".join(SKIP_EXECUTION_PATTERNS))

FENCE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_markdown_files():
    return sorted(REPO_ROOT.rglob("*.md"))


def extract_snippets(md_path: Path):
    """Return list of (heading_context, code_str) tuples."""
    text = md_path.read_text(encoding="utf-8")
    snippets = []
    last_heading = "(top-level)"
    for line in text.split("\n"):
        if line.startswith("#"):
            last_heading = line.strip()
    # Re-do properly: track heading as we scan
    current_heading = "(top-level)"
    pos = 0
    for m in FENCE_RE.finditer(text):
        # Find last heading before this match
        before = text[:m.start()]
        heading_matches = list(re.finditer(r"^#+.+", before, re.MULTILINE))
        if heading_matches:
            current_heading = heading_matches[-1].group().strip()
        snippets.append((current_heading, m.group(1)))
    return snippets


def should_skip_execution(code: str) -> bool:
    return bool(_SKIP_RE.search(code))


def syntax_check(code: str) -> tuple[bool, str]:
    try:
        compile(code, "<snippet>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def execute_snippet(code: str) -> tuple[bool, str]:
    ns = {}
    try:
        exec(compile(code, "<snippet>", "exec"), ns)  # noqa: S102
        return True, ""
    except Exception:
        return False, traceback.format_exc(limit=5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    md_files = find_markdown_files()
    total = 0
    passed = 0
    skipped = 0
    failed = 0
    failures = []

    for md_path in md_files:
        rel = md_path.relative_to(REPO_ROOT)
        snippets = extract_snippets(md_path)
        if not snippets:
            continue

        for i, (heading, code) in enumerate(snippets):
            total += 1
            label = f"{rel} [{heading}] snippet #{i+1}"

            # 1. Syntax check (always)
            ok, err = syntax_check(code)
            if not ok:
                failed += 1
                failures.append((label, "SYNTAX", err, code))
                print(f"  FAIL  {label}")
                print(f"        {err}")
                continue

            # 2. Execution check (only for pure-logic blocks)
            if should_skip_execution(code):
                skipped += 1
                print(f"  SKIP  {label}")
                continue

            ok, err = execute_snippet(code)
            if ok:
                passed += 1
                print(f"  PASS  {label}")
            else:
                failed += 1
                failures.append((label, "RUNTIME", err, code))
                print(f"  FAIL  {label}")
                # Print first 3 lines of traceback
                for line in err.strip().split("\n")[:4]:
                    print(f"        {line}")

    print()
    print("=" * 70)
    print(f"Results: {total} snippets | {passed} PASS | {skipped} SKIP | {failed} FAIL")
    print("=" * 70)

    if failures:
        print("\nFailed snippets:\n")
        for label, kind, err, code in failures:
            print(f"--- {kind}: {label} ---")
            print(err.strip())
            print()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
