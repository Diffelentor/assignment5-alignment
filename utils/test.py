from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
generated_text="To compute \( \frac{1}{31} \div \frac{1}{93} \), we can use the property of division of fractions which states that dividing by a fraction is equivalent to multiplying by its reciprocal. So, \( \frac{1}{31} \div \frac{1}{93} = \frac{1}{31} \times \frac{93}{1} = \frac{93}{31} \). Now, we need to simplify \(\frac{93}{31}\). Since 93 is divisible by 31, we can simplify it to 3. So, the answer is 3. </think> <answer> 3 </answer>"
gt="3"
rewards = r1_zero_reward_fn(generated_text, gt)
print(rewards)
fmt_reward = rewards.get("format_reward", 0)
ans_reward = rewards.get("answer_reward", 0)
print(fmt_reward, ans_reward)