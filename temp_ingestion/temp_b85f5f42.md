[[Problems/Full Text Justification]]
### Loop Invariant / Fighting the Iterator
- A loop invariant is a property that is true before the loop starts, true at the start of every iteration, and true after the loop ends.
- Try to keep the invariant (typically the i & j in loop definitions) stable.
- If you're going to modify them, avoid having to "overshoot, then compensate" behaviour and instead handle it in well defined phases. 
- **Rule of Thumb:** If you find yourself writing i-- or r-- inside a for loop, you are using the wrong pattern. Switch to a while loop with a lookahead.[[ Sliding Window]]

### Orthogonal Separation
- Try to combat the tendency to do premature optimisation by clubbing everything in a single scope/pass. Create branching/modularise for logic that can obviously be separated.

### Math then Code
- Create & calculate variables with appropriate names prior to entering any kind of loop/code. 
- Typical tendency is to do something like `container\[a\*b+c%k]`. Instead create 
  `var = a\*b+c%k` first.
- This is good practice for real world coding, and also reduces *side effects* (ripple effect) when modifying parts of the code by making code more readable.

### [[Sentinel Values]]
- Instead of writing complex if/else logic for edge cases (like the first node of a linked list or the start of a buffer), you create a "fake" element to make the general math work for everyone. 

### Guard Clause
- The main logic should always reside at the minimum possible indentation level of its scope.
- Helps keep *Mental Stack*(of the programmer/reader) manageable.
  ```cpp
  // ❌ Pattern: The logic is a "guest" inside the condition
void verify() {
    if (valid) {
        // [Happy Path is indented] -> Mental Stack Depth: 1
        save();
    }
}
// ✅ Pattern: The logic is the "owner" of the function
void verify() {
    if (!valid) return;
    
    // [Happy Path is Level 0] -> Mental Stack Depth: 0
    save();
}
  ```