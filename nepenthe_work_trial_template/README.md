# Nepenthe Unlearning Work Trial

Hi there, and welcome!

We're really excited to have you join us for this work trial. The goal of the Nepenthe project is ambitious: to develop the first truly deep and robust machine unlearning methods. We believe this is a critical component for building safe and controllable AI, and we're looking for the best partners to help us tackle this challenge.

This short, paid work trial is designed to be a two-way street. It allows us to see how your team approaches a realistic engineering problem, and it gives you a chance to get a feel for the kinds of challenges we're working on. We're not looking for a perfect, state-of-the-art result. We're far more interested in your thought process, the clarity of your code, and how you work together as a team.

Let's get started.

---

## 1. The Goal of This Work Trial

Over the next 1-3 days, your goal is to **implement a novel unlearning technique within our existing evaluation framework.**

This task is designed to test a few key things:
*   **Codebase Navigation:** How quickly can you get acquainted with a complex, research-oriented codebase?
*   **Technical Implementation:** Can you translate a high-level method specification into a working implementation?
*   **Problem-Solving:** How do you approach the inevitable small hurdles and ambiguities that come up during development?

Think of this as the first week on a new project. We want to see how you dive in.

---

## 2. Getting Started: Codebase Overview

You've been provided with our evaluation codebase, which is used to run unlearning experiments and then test their robustness using the "Retraining on T" (RTT) attack proposed by Deeb & Roger (2024).

### Key Files
*   `conf/`: **This is your control panel.** It contains all experiment configurations using the [Hydra](https://hydra.cc/) framework. You'll spend a lot of time here.
*   `pipeline.py`: **The central orchestrator.** This script reads your Hydra config, sets up experiments, and dispatches them as parallel jobs using `ray`.
*   `unlearn_corpus.py`: Implements several "standard" unlearning methods (like Gradient Descent).
*   `finetune_corpus.py`: Implements the RTT fine-tuning attack. This is the "adversary" that tries to undo the unlearning.
*   `rmu/unlearn_pipeline.py`: Contains the implementation for a more advanced method, Representation Misdirection for Unlearning (RMU).

### The Main Experiment Workflow

Understanding this flow is key to understanding the codebase:

1.  **Configuration (`conf/`):** An experiment begins when you run `python pipeline.py`. Hydra immediately loads a configuration file (e.g., `conf/default.yaml`) that specifies everything: the base model, the datasets, and the unlearning method to use along with its hyperparameters.

2.  **Orchestration (`pipeline.py`):** The main `run_pipeline` function parses the config and launches a series of parallel tasks using Ray. The primary task is defined in the `main` function within `pipeline.py`.

3.  **The `unlearn` Router:** The `main` function first calls the `unlearn` function, which acts as a router. Based on the `unlearn_type` you set in the config, it delegates the unlearning task to the correct implementation (e.g., `unlearn_corpus.main` or `rmu.unlearn_pipeline.main`). This step takes a base model and produces a new, "unlearned" model.

4.  **The `RTT` Evaluation:** After a model has been unlearned, the `main` function in `pipeline.py` launches a new set of tasks using `finetune_corpus.main`. This script fine-tunes the unlearned model on a portion of the forgotten data to see if the knowledge can be recovered.

---

## 3. The Task: Implementing "Sequential Unlearning"

Your core task is to implement a new meta-algorithm we'll call **Sequential Unlearning**. The intuition is that applying unlearning in a single shot can create a simple, brittle defense. Sequential Unlearning aims to create a more robust, "defense-in-depth" solution by building up defenses in distinct stages.

Instead of a single unlearning step, you will implement a wrapper that applies a **base unlearning method** repeatedly in a specific sequence.

### Method Specification

The Sequential Unlearning algorithm takes a set of data to be forgotten, partitions it into `k` disjoint folds `F_1, F_2, ..., F_k`, and then proceeds in `k` stages. The model produced at the end of one stage becomes the input model for the next.

For each stage `i` from 1 to `k`, you will:

1.  **Define a Cumulative `ForgetTarget_i`:** This is the set of all data to be forgotten up to the current stage.
    *   `ForgetTarget_i = F_1 ∪ F_2 ∪ ... ∪ F_i`

2.  **Define a Dynamic `RetainTarget_i`:** This is the set of data the model must *not* forget at this stage. Crucially, it includes the standard "retain set" (e.g., general knowledge from MMLU) **plus all future forget folds**.
    *   `RetainTarget_i = R_initial ∪ F_{i+1} ∪ ... ∪ F_k`

3.  **Apply the Base Unlearning Method:** Call a standard unlearning method `U` using the model from the previous stage (`θ_{i-1}`) and the two sets defined above.
    *   `θ_i = U(θ_{i-1}, ForgetTarget_i, RetainTarget_i)`

The final model, `θ_k`, is the output of the last stage.

**A Concrete Example (k=3):**

Let's say your `forget_data` is split into folds `F1`, `F2`, `F3`, and you have an initial retain set `R`.

*   **Stage 1:** Unlearn `F1` while retaining `F2`, `F3`, and `R`.
    *   `ForgetTarget = {F1}`
    *   `RetainTarget = {F2, F3, R}`
    *   Run the base unlearning method on these sets.

*   **Stage 2:** Take the model from Stage 1. Now, unlearn `{F1, F2}` while retaining `F3` and `R`.
    *   `ForgetTarget = {F1, F2}`
    *   `RetainTarget = {F3, R}`
    *   Run the base unlearning method.

*   **Stage 3:** Take the model from Stage 2. Unlearn `{F1, F2, F3}` while retaining only `R`.
    *   `ForgetTarget = {F1, F2, F3}`
    *   `RetainTarget = {R}`
    *   Run the base unlearning method.

For your implementation, you should use the existing **RMU (`UnlearnType.CUT`)** as your base unlearning method `U`.

---

### 4. Your Task & Deliverables

Here’s a checklist of what we'd like to see. We’ve structured this to understand both *what* you built and *how* you built it.

1.  **Implement `SequentialUnlearning`:** Create a new Python module (e.g., `sequential_unlearning.py`) that contains the logic for the sequential loop described above.

2.  **Integrate Your Method:** Update `pipeline.py` and any other necessary files to integrate your new method.

3.  **Create a Configuration:** Add a new Hydra config file in the `conf/` directory that runs your method. Think about what new parameters it might require (e.g., number of folds).

4.  **Share Your Progress (Daily Updates):** We'd love to follow along with your process. Please provide a short progress update at the end of each day. This can be a brief text summary or a quick Loom video (whichever you prefer). We're especially interested in:
    *   What you accomplished that day.
    *   Any challenges you faced (e.g., interpreting the spec, navigating the codebase).
    *   Any assumptions you made or solutions you devised.

5.  **The Final Deliverable (Code + Write-up):** To wrap things up, please send us a single `.zip` file or a link to a private GitHub repository containing:
    *   **Your modified codebase.**
    *   **A brief final write-up (1-2 pages is perfect).** This helps us understand the "why" behind your code. Please include:
        *   A summary of your implementation and how it maps to the spec.
        *   A quick overview of the experiment run(s) you kicked off (no need to wait for full results, just what you configured and launched).
        *   Any notable observations, design decisions, or ideas for improvement you might have.

This structure gives us a complete picture of your work and is exactly what we'd look for in a long-term collaborator.

---

## 5. Evaluation

Again, we want to be very clear: **this is not a test with a single right answer.** We are not grading you on whether your implementation achieves a specific accuracy.

We will be reviewing your submission qualitatively, looking for:

*   **Code Quality:** Is the code clean, readable, and well-structured?
*   **Design Choices:** How did you decide to structure your new module and its integration?
*   **Pragmatism:** How did you manage the complexity of the task within the time limit?

This is our way of simulating a real, collaborative project sprint.

Thank you again for your time and effort. We genuinely appreciate it and are looking forward to seeing what you build.

Good luck!