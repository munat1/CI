#!/usr/bin/python

# Copyright 2018 Malte Klaassen, Witali Rau
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np

# Activation function is a simple x>=1?1:0
f_act = lambda x: 1 if x >= 1 else 0

# Initial State of the RNN is all 0s
state = np.zeros((3,1))

# We use a RNN w/o bias w/ 3 hidden units and 2 inputs, so every hidden unit needs 3 + 2 weights
weights_hidden = np.array(
        [
            [1,     1,      0,      1,      0], # >= 1 on previous bits + carry
            [0.5,   0.5,    0,      0.5,    0], # >= 2
            [0.34,  0.34,   0,      0.34,   0]  # >= 3
        ])

# and the output 3 weights
weights_output = np.array([1, -1, 1])

def g(x, y):
    global state
    state = np.zeros((3,1))
    output = []
    for t in range(0, len(x)+2):
        print(str(t) + ":")
        print("Current state: \n" + str(state))
        newstate = np.zeros((3,1))
        ins = np.vstack((np.array([[np.hstack((x,[0,0]))[t]], [np.hstack((y,[0,0]))[t]]]), state))
        print("\"Input\" (including old state):\n" + str(ins))
        for i in range(0, 3):
            acc = np.dot(weights_hidden[i], ins)[0]
            print("a" + str(i) + ": " + str(acc) + ", o" + str(i) + ": " + str(f_act(acc)))
            newstate[i][0] = f_act(acc)
        out = np.dot(weights_output, state)[0]
        print("Out: " + str(out))

        state = newstate
        output.append(out)

        print("\n")
    return output[1:] # Drop the first output bit as it doesn't belong to the result

def tobin(x, pad=None):
    o = []
    while x:
        if x % 2 == 1:
            o.append(1)
        else:
            o.append(0)
        x = np.floor(x/2)
    if pad is not None:
        while len(o) < pad:
            o.append(0)
    return o

def frombin(bs):
    f = 1
    o = 0
    for b in bs:
        o = o + b * f
        f = f * 2
    return o

x = np.random.randint(0, 2**9 - 1)
y = np.random.randint(0, 2**9 - 1)
bx = tobin(x, pad=9)
by = tobin(y, pad=9)

bz  = g(bx, by)
z = frombin(bz)

print("%s + %s = %s" % (x, y, z))
print("Binary (LSB): %s + %s = %s" % (bx, by, bz))


