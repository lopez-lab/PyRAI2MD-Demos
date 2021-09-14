import json,sys

def format(x):
    output=''
    for i in x:
        if   len(i) == 4:
            output+='%-5s%24.16f%24.16f%24.16f\n' % (i[0],float(i[1]),float(i[2]),float(i[3]))
        elif len(i) == 3:
       	    output+='%24.16f%24.16f%24.16f\n' % (float(i[0]),float(i[1]),float(i[2]))
    return output

with open(sys.argv[1],'r') as indata:
    data=json.load(indata)

    natom,nstate,xyzset,invrset,energyset,gradset,nacset,civecset,movecset=data

outtext="""Training data for NNs in PyRAIMD
===========================================================
Atoms:     %d
States:    %d
Points:    %d
===========================================================
""" % (natom, nstate, len(xyzset))
for n,i in enumerate(xyzset):

    s0,s1=energyset[n]
    g0,g1=gradset[n]
    nac=nacset[n][0]
    outtext+='Points %s [Angstrom]\n%s'  % (n+1,format(i))
    outtext+='Energy [Hartree]\n%24.16f%24.16f\n'  % (float(s0),float(s1))
    outtext+='Force 0 [Hartree/Bohr]\n%sForce 1 [Hartree/Bohr]\n%s' % (format(g0),format(g1))
#    outtext+='NAC(interstate) 0 - 1 [Hartree/Bohr]\n%s'  % (format(nac))


with open('%s.txt' % (sys.argv[1].split('.')[0]),'w') as outfile:
    outfile.write(outtext)
