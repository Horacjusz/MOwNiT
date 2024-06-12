from copy import deepcopy

q0 = "q0"
q1 = "q1"
q2 = "q2"
q3 = "q3"
T = "T"
a = "a"
b = "b"
c = "c"
e = "e"
s = "#"


def stack_automa(init_word, stack_start) :
    state, stack = stack_start

    init_word += e
    
    index = 0

    scanned = ""

    letter = init_word[0]
    
    while stack :
        # print(len(stack))
        if state == q0 :
            if letter == a :
                stack.append(s)
                stack.append(s)
                scanned += letter
            elif letter == b :
                state = q1
                stack.pop()
                scanned += letter
            elif letter == c : 
                state = T
                scanned += letter
            elif letter == e : 
                state = q3
        elif state == q1 :
            if letter == a :
                state = T
                scanned += letter
            elif letter == b :
                state = q1
                stack.pop()
                scanned += letter
            elif letter == c : 
                state = q2
                stack.pop()
                scanned += letter
            elif letter == e :
                state = q3
        elif state == q2 :
            if letter == a :
                state = T
                scanned += letter
            elif letter == b :
                state = T
                scanned += letter
            elif letter == c : 
                state = q2
                stack.pop()
                scanned += letter
            elif letter == e :
                state = q3
                 
        if state == q3 :
            if len(stack) <= 1 :
                # print(scanned, "is recognized by automa")
                return True
            state = T
        if state == T :
            # print(scanned, "is not recognized by automa")
            return False
        
        index += 1
        letter = init_word[index]
                
    print("Error")
    return
            

# stack_automa("aaabbbccc", (q0, [s]))

for k in range(70) :
    word = ""
    for i in range(k) :
        word += a
        word_a = deepcopy(word)
        for j in range(i + 1) :
            word += b
            word_b = deepcopy(word)
            for l in range(j + 1) :
                word += c
                # if i == j == l : print(">>>>>>> ", end = "")
                val = stack_automa(word, (q0, [s]))
                # print(word)
                if i == j == l and not val : print("=================================================ERROR=================================================")
                if not (i == j == l) and val : print("=================================================ERROR=================================================")
            word = word_b
        word = word_a
    print(k)