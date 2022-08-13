import os
import xml.etree.ElementTree as elemTree
import math
import time

joint_stack = []

def rad2deci2string(a):
    b = a.split()
    return (
        str(math.radians(float(b[0]))) + 
        " " + 
        str(math.radians(float(b[1])))
    )

def tung(a):
    global joint_stack
    for d in a.findall('joint'):
        joint_stack.append([d.tag, d.attrib])
    for d in a.findall('body'):
        tung(d)

root = elemTree.parse(os.getcwd()+'/humanoid.xml').getroot()

for d in root.find('worldbody'):
    tung(d)

for d in joint_stack:
    print(
        '<position '+
        'ctrllimited="true" '+
        'kp="90" '+
        'ctrlrange="%s" ' %  rad2deci2string(d[1]['range'])+
        'joint="%s" ' %  d[1]['name']+
        'name="%s"/>' %  d[1]['name']
    )

for d in joint_stack:
    print(
        '<velocity '+
        'ctrllimited="true" '+
        'kp="90" '+
        'ctrlrange="%s" ' %  rad2deci2string(d[1]['range'])+
        'joint="%s" ' %  d[1]['name']+
        'name="%s"/>' %  d[1]['name']
    )

time.sleep(10000)