# Sample Conversations for Context Testing

This document contains sample multi-turn conversations to test whether the ACE chatbot maintains context across messages. Each conversation is designed to verify that the bot remembers previous exchanges.

---

## Test 1: Basic Math Context

**Purpose**: Test if bot remembers numerical results from previous calculations

1. **User**: "What is 15 + 27?"
   - **Expected**: Bot calculates and returns 42

2. **User**: "Now multiply that number by 3"
   - **Expected**: Bot should remember 42 and return 126

3. **User**: "What's half of the previous result?"
   - **Expected**: Bot should remember 126 and return 63

4. **User**: "Add 37 to it"
   - **Expected**: Bot should remember 63 and return 100

**✅ Pass Criteria**: Bot correctly references previous calculations without needing to repeat the numbers

---

## Test 2: Information Recall

**Purpose**: Test if bot remembers information provided earlier in conversation

1. **User**: "My name is Alice and I work as a software engineer in Seattle"
   - **Expected**: Bot acknowledges the information

2. **User**: "What's my name?"
   - **Expected**: Bot should respond "Alice"

3. **User**: "What do I do for work?"
   - **Expected**: Bot should respond "You're a software engineer"

4. **User**: "Where do I work?"
   - **Expected**: Bot should respond "Seattle"

**✅ Pass Criteria**: Bot accurately recalls all three pieces of information without confusion

---

## Test 3: List Building

**Purpose**: Test if bot can build upon lists created in previous messages

1. **User**: "Let's create a shopping list. Add milk and eggs"
   - **Expected**: Bot creates a list with milk and eggs

2. **User**: "Add bread and butter to the list"
   - **Expected**: Bot adds to the existing list (now has 4 items)

3. **User**: "Remove eggs from the list"
   - **Expected**: Bot removes eggs (now has milk, bread, butter)

4. **User**: "How many items are on the list now?"
   - **Expected**: Bot should answer "3 items"

5. **User**: "What's on the list?"
   - **Expected**: Bot should list milk, bread, and butter

**✅ Pass Criteria**: Bot maintains the list state across all modifications

---

## Test 4: Story Continuation

**Purpose**: Test if bot can continue a narrative coherently

1. **User**: "Let's write a story. Start with: 'Once upon a time, there was a brave knight named Sir Roland'"
   - **Expected**: Bot starts the story

2. **User**: "What happens next? Add a dragon to the story"
   - **Expected**: Bot continues the story about Sir Roland and introduces a dragon

3. **User**: "How does Sir Roland defeat the dragon?"
   - **Expected**: Bot continues the same story with a resolution

4. **User**: "What was the knight's name again?"
   - **Expected**: Bot should remember "Sir Roland"

**✅ Pass Criteria**: Bot maintains story coherence and character consistency

---

## Test 5: Comparative Analysis

**Purpose**: Test if bot remembers multiple items for comparison

1. **User**: "Tell me about Python programming language"
   - **Expected**: Bot provides information about Python

2. **User**: "Now tell me about JavaScript"
   - **Expected**: Bot provides information about JavaScript

3. **User**: "Which one is better for web development?"
   - **Expected**: Bot compares both languages mentioned earlier

4. **User**: "What about the first language we discussed - is it object-oriented?"
   - **Expected**: Bot refers back to Python and discusses its OOP features

**✅ Pass Criteria**: Bot maintains context of both programming languages and can compare them

---

## Test 6: Sequential Instructions

**Purpose**: Test if bot can follow multi-step instructions that depend on previous steps

1. **User**: "Create a workout plan for Monday: 30 minutes running"
   - **Expected**: Bot creates Monday's plan

2. **User**: "For Tuesday, add weightlifting for 45 minutes"
   - **Expected**: Bot adds Tuesday to the existing plan

3. **User**: "What's the total workout time for the week so far?"
   - **Expected**: Bot should calculate 30 + 45 = 75 minutes

4. **User**: "Add Wednesday: 20 minutes yoga"
   - **Expected**: Bot adds Wednesday

5. **User**: "Show me the complete weekly plan"
   - **Expected**: Bot lists Monday (running), Tuesday (weightlifting), Wednesday (yoga)

**✅ Pass Criteria**: Bot maintains the entire workout plan across all additions

---

## Test 7: Preference Memory

**Purpose**: Test if bot remembers user preferences stated earlier

1. **User**: "I'm allergic to peanuts and I don't like spicy food"
   - **Expected**: Bot acknowledges the preferences

2. **User**: "Suggest a meal for dinner"
   - **Expected**: Bot suggests something without peanuts and not spicy

3. **User**: "What about a dessert?"
   - **Expected**: Bot suggests a dessert without peanuts

4. **User**: "What were my dietary restrictions again?"
   - **Expected**: Bot recalls peanut allergy and dislike of spicy food

**✅ Pass Criteria**: Bot consistently remembers and applies user preferences

---

## Test 8: Problem-Solving Context

**Purpose**: Test if bot maintains context during problem-solving

1. **User**: "I need to plan a road trip from New York to Los Angeles"
   - **Expected**: Bot acknowledges the trip details

2. **User**: "The trip is 2,800 miles. If I drive 400 miles per day, how many days will it take?"
   - **Expected**: Bot calculates 7 days

3. **User**: "If I want to complete it in 5 days instead, how many miles per day?"
   - **Expected**: Bot remembers the 2,800 miles and calculates 560 miles/day

4. **User**: "Where am I traveling from and to?"
   - **Expected**: Bot should remember New York to Los Angeles

**✅ Pass Criteria**: Bot maintains trip details and calculations throughout

---

## Test 9: Conversation Topic Switch and Return

**Purpose**: Test if bot can handle topic switches while maintaining context

1. **User**: "Tell me about the solar system"
   - **Expected**: Bot provides information about the solar system

2. **User**: "Actually, let's talk about cooking instead. What's a good recipe for pasta?"
   - **Expected**: Bot switches to cooking topic

3. **User**: "How many planets did we discuss earlier?"
   - **Expected**: Bot should return to solar system context and mention 8 planets

4. **User**: "And what were we discussing about food?"
   - **Expected**: Bot should remember the pasta recipe discussion

**✅ Pass Criteria**: Bot maintains both conversation threads and can switch between them

---

## Test 10: Complex Playbook Building

**Purpose**: Test how playbook bullets accumulate and are used in context

1. **User**: "Plan a birthday party for 20 people with a budget of $500"
   - **Expected**: Bot creates a plan and extracts bullets about party planning

2. **User**: "Now plan a wedding for 100 people with a $10,000 budget"
   - **Expected**: Bot should use learnings from first event, mentions scale differences

3. **User**: "What did we plan first - remind me of the budget?"
   - **Expected**: Bot remembers the birthday party with $500 budget

4. **User**: "What strategies would work for both events?"
   - **Expected**: Bot uses accumulated playbook bullets to suggest common strategies

**✅ Pass Criteria**: Bot maintains context AND applies learned playbook bullets across conversations

---

## Test 11: Pronoun Resolution

**Purpose**: Test if bot correctly resolves pronouns based on context

1. **User**: "Tell me about Albert Einstein"
   - **Expected**: Bot provides information about Einstein

2. **User**: "What was his most famous theory?"
   - **Expected**: Bot understands "his" refers to Einstein, mentions relativity

3. **User**: "When did he publish it?"
   - **Expected**: Bot remembers both Einstein and the theory, provides date (1905/1915)

4. **User**: "Where was he born?"
   - **Expected**: Bot remembers referring to Einstein, answers Germany

**✅ Pass Criteria**: Bot correctly resolves all pronouns to Einstein

---

## Test 12: Conditional Logic Memory

**Purpose**: Test if bot remembers conditional statements

1. **User**: "If it rains tomorrow, I'll stay home and read. If it's sunny, I'll go hiking"
   - **Expected**: Bot acknowledges both conditions

2. **User**: "What will I do if it rains?"
   - **Expected**: Bot recalls "stay home and read"

3. **User**: "What about if the weather is nice?"
   - **Expected**: Bot recalls "go hiking"

4. **User**: "What were the two options I mentioned?"
   - **Expected**: Bot recalls both conditions and activities

**✅ Pass Criteria**: Bot remembers both conditional branches accurately

---

## How to Use These Tests

1. **Run tests in sequence**: Copy each user message in order and paste into the chat
2. **Clear chat between test suites**: Use "Clear Chat History" button between each Test (#1-12)
3. **Check responses**: Verify bot responses match expected behavior
4. **Note failures**: Document any instances where context is lost
5. **Check playbook**: After several tests, verify playbook bullets are being created and used

## Success Metrics

- **Context Retention**: Bot maintains information for 4+ turns
- **Accurate Recall**: Bot retrieves correct information when asked
- **Pronoun Resolution**: Bot correctly identifies referents
- **Playbook Integration**: Bot learns from conversations and applies bullets
- **No Hallucination**: Bot doesn't invent information not provided

## Known Limitations to Test

1. **Very long conversations**: Test with 10+ turns to see if context window fills
2. **Complex nested context**: Multiple topics interleaved
3. **Ambiguous references**: Test with unclear pronouns
4. **Contradictory information**: Provide conflicting info to see how bot handles it

---

## Additional Edge Cases

### Test 13: Number Sequences

1. "Remember this sequence: 2, 4, 8, 16"
2. "What comes next in the sequence?"
3. "What was the first number?"
4. "How many numbers did I give you?"

### Test 14: Time-Based Context

1. "I have a meeting at 2 PM today"
2. "I also have dinner plans at 7 PM"
3. "What's my first commitment?"
4. "How much time do I have between my two events?"

### Test 15: Multiple Entity Tracking

1. "John is 25 years old and works as a teacher"
2. "Mary is 30 years old and works as a doctor"
3. "How old is John?"
4. "What does Mary do?"
5. "Who is older?"

---

**Last Updated**: October 11, 2025
**Purpose**: Validate conversation context maintenance in ACE Streamlit Demo
