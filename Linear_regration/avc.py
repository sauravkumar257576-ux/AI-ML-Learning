import turtle
t = turtle.Turtle()
s = turtle.Screen()
s.bgcolor("black")
t.speed(8)
turtle.tracer(3,0)
t.color("red")

for i in range(400):
    t.forward(i)
    t.left(125)
    t.forward(i)
    t.left(45)
turtle.done()