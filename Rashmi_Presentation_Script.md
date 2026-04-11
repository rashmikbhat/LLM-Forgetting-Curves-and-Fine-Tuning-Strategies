# Presentation Script: Catastrophic Forgetting in Fine-Tuned LLMs

**Total Duration: ~5-6 minutes**

---

## SLIDE 1: Introduction (60-75 seconds)

Good morning everyone. Today we're going to present our research on catastrophic forgetting in fine-tuned large language models.

So first, what is catastrophic forgetting? When we fine-tune pre-trained LLMs on specific tasks, we see two things happen. The good news is that task performance improves significantly. But the bad news is that the model's general knowledge degrades—specifically, perplexity increases. Essentially, the model "forgets" the capabilities it learned during pre-training.

Let us give you a concrete example. We took GPT-2 and measured its baseline perplexity on WikiText-2, which is general Wikipedia text. It scored 29.4. After fine-tuning on SST-2, a sentiment classification task, the perplexity jumped to 35.8. That's a 22% degradation in the model's ability to handle general text.

Now, there's a significant research gap here. We have three critical unknowns. First, which PEFT method—parameter-efficient fine-tuning method—actually minimizes forgetting? Second, what hyperparameters should we use to optimize knowledge retention? And third, can we use mitigation strategies to prevent this degradation altogether?

One important note about our evaluation: we're measuring general language forgetting using WikiText-2 perplexity. This isn't about task-specific capability gaps or transfer learning between different tasks. We're measuring how much genuine general knowledge the model loses.

Our research design consists of four complementary studies. Study 1 establishes the forgetting bounds by comparing full fine-tuning against frozen models. Study 2 explores LoRA with rank ablation—testing ranks 4, 8, and 16. Study 3 does the same for Prefix Tuning with different prefix lengths. And Study 4 tests adapters combined with replay, which is a memory rehearsal technique using different replay ratios.

What makes our study unique is that it's the first unified comparison of PEFT methods specifically focused on forgetting. We ran systematic hyperparameter ablation studies. We tested an active mitigation strategy—this replay technique. And we validated our findings across three different task types.


---

## SLIDE 2: Experimental Results (90-120 seconds)

So for this study, we focused on testing whether memory rehearsal—specifically mixing WikiText-2 general text with task data during fine-tuning—can prevent catastrophic forgetting. We're using adapters, which are small trainable modules—only 64,000 parameters—that's just 0.02% of the base GPT-2 Medium model.

Phase 1 was a replay ratio ablation study using SST-2 and GPT-2 Medium. The research question was straightforward: does mixing general text actually prevent forgetting?

Let us walk you through the results. Our baseline model—GPT-2 Medium without any fine-tuning—had a WikiText-2 perplexity of 48.9. When we fine-tuned with zero percent replay—meaning 100% task data—perplexity exploded to 101. That's a massive 106% increase, showing clear catastrophic forgetting.

Now here's where it gets interesting. With 10% replay, the perplexity dropped dramatically to 36.4. That's actually 26% better than the baseline—not just preventing forgetting, but improving on the original model. And with 20% replay, we achieved 32.9 perplexity, which is 33% better than baseline.

Let us highlight the key metrics here. At zero percent replay, we saw 106% forgetting compared to baseline. With 20% replay, we achieved 67% recovery from that catastrophic forgetting. And the final result is 33% better perplexity than the pre-trained baseline.

The key finding from Phase 1 is that 20% replay doesn't just prevent forgetting—it actually improves the model's general language capabilities beyond the original baseline.

Now, Phase 2 tested whether this generalizes across different tasks. We took our best replay ratio from Phase 1—which was 10%—and tested it on two additional tasks: SNLI, which is natural language inference, and SQuAD, which is question answering.

Looking at the table, SST-2 without replay had a perplexity of 101. With 10% replay, it dropped to 36.4—that's 64% recovery. SNLI showed the worst forgetting—perplexity jumped all the way to 123 without replay—but also the best recovery at 70% with 10% replay. SQuAD was the least affected, showing only 8% forgetting initially, and achieving 34% recovery.

The consistency is remarkable. Across all three tasks, we're seeing 34 to 70% recovery. And all tasks with 10% replay achieve perplexity in the 34 to 37 range, which is near or better than the pre-trained baseline. This is especially impressive given that adapters only use 64,000 parameters—that's just 0.02% of the base model.

The graph here shows the forgetting curves with the baseline. You can clearly see how perplexity shoots up during fine-tuning without replay, but stays controlled with replay.

---

## SLIDE 3: Analysis (90-120 seconds)

Now let's analyze why replay works and what the practical implications are.

The replay mechanism is essentially memory rehearsal. Without replay, you're training on 100% task-specific data, which leads to catastrophic forgetting with perplexity increasing anywhere from 7% to 151% depending on the task. With 10% replay, you're mixing 90% task data with 10% WikiText-2, which gives you balanced learning and keeps perplexity in the 34 to 37 range, near the baseline. With 20% replay, you're doing 80% task and 20% general text, which provides maximum protection and actually achieves perplexity of 32 to 33—which is 33% better than baseline.

Looking at the task distance correlation, we see interesting patterns. SST-2 showed 106% forgetting without replay, and achieves 64 to 67% recovery with replay. SNLI showed the worst forgetting at 151%, but also the best recovery at 70 to 74%. And SQuAD showed minimal forgetting at only 8%, with 34 to 39% recovery. The pattern here is clear: SNLI shows the worst forgetting but benefits most from replay, while SQuAD is least affected by adapter fine-tuning.

In terms of configuration trade-offs, you have two good options. Ten percent replay gives you perplexity in the 34 to 37 range with about 64% recovery and minimal overhead. This is perfect for balanced deployments. Twenty percent replay gives you 32 to 33 perplexity with 67% recovery and is best when retention is critical.

Let us highlight the key discoveries. First, replay is highly effective—we're seeing 34 to 74% recovery from catastrophic forgetting. Twenty percent replay doesn't just prevent forgetting; it achieves 33% better perplexity than the baseline. And it's simple to implement.

Second, the optimal configuration depends on your needs. Use 20% replay for maximum retention, 10% replay for a balanced approach. Both beat the baseline.

Third, we're seeing task-specific patterns. SNLI has the worst forgetting at 151% but the best recovery at 74%. SQuAD shows minimal forgetting at just 8%. SST-2 sits in the middle with 106% forgetting and 67% recovery.

When should you use replay? Use it when you're deploying for multiple purposes, when retention is critical, or when you have access to WikiText-2 or similar general text. Avoid it when you're deploying for a single task only, when performance is absolutely critical, or when you have limited compute resources.

The implementation is straightforward. First, select your replay ratio—10% for balanced, 20% for maximum retention. Second, mix WikiText-2 with your task data during training. Third, monitor both WikiText-2 perplexity and your task metrics. And fourth, validate on a held-out general test set.

---

## SLIDE 4: Model Scaling (60-75 seconds)

Finally, let's look at model scaling and how model size impacts forgetting.

For this scaling experiment, we tested two models. GPT-2 Medium has 345 million parameters. Qwen-1.5B has 1.5 billion parameters—that's 4.3 times larger. We used the same adapter setup with 64,000 parameters, the same replay ratios of 0%, 10%, and 20%, and the same three tasks.

The results show Qwen-1.5B has a massive advantage. Looking at the baseline, Qwen scores 25.9 compared to GPT-2's 48.9—that's 47% better. With 20% replay on SST-2, Qwen achieves 20.3 versus GPT-2's 32.9—38% better. On SNLI, Qwen gets 19.5 versus GPT-2's 32.5—40% improvement. And on SQuAD, it's 17.5 versus 32.2—46% improvement. The improvement is consistent across all tasks at 38 to 46%.

The key insight here is that Qwen-1.5B achieves 17 to 20 perplexity across all tasks with 20% replay, compared to GPT-2's 32 to 33.

The scaling law visualization shows this clearly on a log scale. Let me walk through the key findings.

First, there's a baseline advantage. Larger models start with better general knowledge. Qwen's baseline is 25.9 compared to GPT-2's 48.9.

Second, model size helps both before and after fine-tuning. Qwen maintains 38 to 46% better perplexity even after replay.

Third, and this is critical: replay is still essential regardless of model size. Even Qwen-1.5B shows dramatic forgetting without replay. On SST-2, it goes from a baseline of 25.9 all the way up to 56.5 without replay, then recovers to 20.3 with 20% replay. So replay remains critical even for large models.

Fourth, the pattern is consistent across all three tasks. Larger models provide better retention, but replay is essential for all model sizes.

---

## CONCLUSION (15-20 seconds)

To summarize, replay is a simple but highly effective technique for preventing catastrophic forgetting in fine-tuned LLMs. Twenty percent replay provides the best retention and actually improves beyond baseline performance. The technique generalizes across different tasks and model sizes. And it's practical to implement with minimal overhead.

Thank you. We're happy to take any questions.

---

## TIMING BREAKDOWN

- Slide 1 (Introduction): 60-75 seconds
- Slide 2 (Results): 90-120 seconds
- Slide 3 (Analysis): 90-120 seconds
- Slide 4 (Scaling): 60-75 seconds
- Conclusion: 15-20 seconds

**Total: 5-6 minutes**

---

## TIPS FOR DELIVERY

1. **Pacing**: Speak at a moderate pace. Don't rush through the numbers.

2. **Emphasis**: When you read numbers like "106% forgetting" or "67% recovery", pause slightly before and after for impact.

3. **Transitions**: The script has natural transitions built in ("Now let's look at...", "Finally...", etc.). Use these to shift slides.

4. **Eye Contact**: Even though you're reading, look up briefly when you say "let us show you" or "looking at the table/graph."

5. **Pointer**: Use a pointer or cursor when you reference specific parts of tables or graphs ("Looking at the table...", "The graph here shows...").

6. **Questions**: At the end, pause for 2-3 seconds before saying "We're happy to take questions" to signal you're done.

7. **Backup Answers**: Be ready to explain:
   - What is WikiText-2? (General Wikipedia text corpus)
   - What are adapters? (Small trainable modules added to frozen models)
   - Why these specific tasks? (SST-2=sentiment, SNLI=inference, SQuAD=QA - covers different domains)
