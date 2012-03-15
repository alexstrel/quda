# -*- coding: utf-8 -*-
import sys

### complex numbers ########################################################################

def complexify(a):
    return [complex(x) for x in a]

def complexToStr(c):
    def fltToString(a):
        if a == int(a): return `int(a)`
        else: return `a`
    
    def imToString(a):
        if a == 0: return "0i"
        elif a == -1: return "-i"
        elif a == 1: return "i"
        else: return fltToString(a)+"i"
    
    re = c.real
    im = c.imag
    if re == 0 and im == 0: return "0"
    elif re == 0: return imToString(im)
    elif im == 0: return fltToString(re)
    else:
        im_str = "-"+imToString(-im) if im < 0 else "+"+imToString(im)
        return fltToString(re)+im_str


### projector matrices ########################################################################

id = complexify([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
])

gamma1 = complexify([
    0,  0, 0, 1j,
    0,  0, 1j, 0,
    0, -1j, 0, 0,
    -1j,  0, 0, 0
])

gamma2 = complexify([
    0, 0, 0, 1,
    0, 0, -1,  0,
    0, -1, 0,  0,
    1, 0, 0,  0
])

gamma3 = complexify([
    0, 0, 1j,  0,
    0, 0, 0, -1j,
    -1j, 0, 0,  0,
    0, 1j, 0,  0
])

gamma4 = complexify([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, -1, 0,
    0, 0, 0, -1
])

igamma5 = complexify([
    0, 0, 1j, 0,
    0, 0, 0, 1j,
    1j, 0, 0, 0,
    0, 1j, 0, 0
])


def gplus(g1, g2):
    return [x+y for (x,y) in zip(g1,g2)]

def gminus(g1, g2):
    return [x-y for (x,y) in zip(g1,g2)]

def projectorToStr(p):
    out = ""
    for i in range(0, 4):
        for j in range(0,4):
            out += complexToStr(p[4*i+j]) + " "
        out += "\n"
    return out

projectors = [
    gminus(id,gamma1), gplus(id,gamma1),
    gminus(id,gamma2), gplus(id,gamma2),
    gminus(id,gamma3), gplus(id,gamma3),
    gminus(id,gamma4), gplus(id,gamma4),
]

### code generation  ########################################################################

### parameters
dagger = False

def block(code):
    lines = ''.join(["    "+line+"\n" for line in code.splitlines()])
    return "{\n"+lines+"}\n"

def sign(x):
    if x==1: return "+"
    elif x==-1: return "-"
    elif x==+2: return "+2*"
    elif x==-2: return "-2*"

def nthFloat4(n):
    return `(n/4)` + "." + ["x", "y", "z", "w"][n%4]

def nthFloat2(n):
    return `(n/2)` + "." + ["x", "y"][n%2]


def in_re(s, c): return "i"+`s`+`c`+"_re"
def in_im(s, c): return "i"+`s`+`c`+"_im"
def g_re(d, m, n): return ("g" if (d%2==0) else "gT")+`m`+`n`+"_re"
def g_im(d, m, n): return ("g" if (d%2==0) else "gT")+`m`+`n`+"_im"
def out1_re(s, c): return "o1_"+`s`+`c`+"_re"
def out1_im(s, c): return "o1_"+`s`+`c`+"_im"
def out2_re(s, c): return "o2_"+`s`+`c`+"_re"
def out2_im(s, c): return "o2_"+`s`+`c`+"_im"
def h1_re(h, c): return ["a","b"][h]+`c`+"_re"
def h1_im(h, c): return ["a","b"][h]+`c`+"_im"
def h2_re(h, c): return ["A","B"][h]+`c`+"_re"
def h2_im(h, c): return ["A","B"][h]+`c`+"_im"
def a_re(b, s, c): return "a"+`(s+2*b)`+`c`+"_re"
def a_im(b, s, c): return "a"+`(s+2*b)`+`c`+"_im"

def tmp_re(s, c): return "tmp"+`s`+`c`+"_re"
def tmp_im(s, c): return "tmp"+`s`+`c`+"_im"


def prolog():
    str = []
    str.append("// *** CUDA DSLASH ***\n\n" if not dagger else "// *** CUDA DSLASH DAGGER ***\n\n")
    str.append("//Extra constant (double) mu, (double)eta and (double)delta\n\n")
    str.append("#define SHARED_FLOATS_PER_THREAD "+`sharedFloats`+"\n\n")
#    str.append("#define SHARED_BYTES_DOUBLE (BLOCK_DIM*SHARED_FLOATS_PER_THREAD*sizeof(double))\n\n")
#    str.append("#define SHARED_BYTES_SINGLE (BLOCK_DIM*SHARED_FLOATS_PER_THREAD*sizeof(float))\n\n")
    
    str.append("// input spinor\n")

    str.append("#ifdef SPINOR_DOUBLE\n")
    str.append("#define spinorFloat double\n")
    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str.append("#define "+in_re(s,c)+" I"+nthFloat2(2*i+0)+"\n")
            str.append("#define "+in_im(s,c)+" I"+nthFloat2(2*i+1)+"\n")
    str.append("\n")
    str.append("#else\n")
    str.append("#define spinorFloat float\n")
    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str.append("#define "+in_re(s,c)+" I"+nthFloat4(2*i+0)+"\n")
            str.append("#define "+in_im(s,c)+" I"+nthFloat4(2*i+1)+"\n")
    str.append("#endif // SPINOR_DOUBLE\n\n")

    str.append("// gauge link\n")

    str.append("#ifdef GAUGE_FLOAT2\n")
    for m in range(0,3):
        for n in range(0,3):
            i = 3*m+n
            str.append("#define "+g_re(0,m,n)+" G"+nthFloat2(2*i+0)+"\n")
            str.append("#define "+g_im(0,m,n)+" G"+nthFloat2(2*i+1)+"\n")

    str.append("// temporaries\n")
    str.append("#define A_re G"+nthFloat2(18)+"\n")
    str.append("#define A_im G"+nthFloat2(19)+"\n")    
    str.append("\n")
    str.append("#else\n")
    for m in range(0,3):
        for n in range(0,3):
            i = 3*m+n
            str.append("#define "+g_re(0,m,n)+" G"+nthFloat4(2*i+0)+"\n")
            str.append("#define "+g_im(0,m,n)+" G"+nthFloat4(2*i+1)+"\n")

    str.append("// temporaries\n")
    str.append("#define A_re G"+nthFloat4(18)+"\n")
    str.append("#define A_im G"+nthFloat4(19)+"\n")    
    str.append("\n")
    str.append("#endif // GAUGE_DOUBLE\n\n")    
            
    str.append("// conjugated gauge link\n")
    for m in range(0,3):
        for n in range(0,3):
            i = 3*m+n
            str.append("#define "+g_re(1,m,n)+" (+"+g_re(0,n,m)+")\n")
            str.append("#define "+g_im(1,m,n)+" (-"+g_im(0,n,m)+")\n")
    str.append("\n")

#no clover term here 

    str.append("// output 1st flavor spinor\n")
    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            if 2*i < sharedFloats:
                str.append("#define "+out1_re(s,c)+" s["+`(2*i+0)`+"*SHARED_STRIDE]\n")
            else:
                str.append("volatile spinorFloat "+out1_re(s,c)+";\n")
            if 2*i+1 < sharedFloats:
                str.append("#define "+out1_im(s,c)+" s["+`(2*i+1)`+"*SHARED_STRIDE]\n")
            else:
                str.append("volatile spinorFloat "+out1_im(s,c)+";\n")
    str.append("\n")

    str.append("// output 2st flavor spinor\n")
    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            if 2*i < sharedFloats:
                str.append("#define "+out2_re(s,c)+" s["+`(2*i+0)`+"*SHARED_STRIDE]\n")
            else:
                str.append("volatile spinorFloat "+out2_re(s,c)+";\n")
            if 2*i+1 < sharedFloats:
                str.append("#define "+out2_im(s,c)+" s["+`(2*i+1)`+"*SHARED_STRIDE]\n")
            else:
                str.append("volatile spinorFloat "+out2_im(s,c)+";\n")
    str.append("\n")
    
    str.append(
"""

#include "read_gauge.h"
#include "read_clover.h"
#include "io_spinor.h"

int sid = blockIdx.x*blockDim.x + threadIdx.x;
int z1 = FAST_INT_DIVIDE(sid, X1h);
int x1h = sid - z1*X1h;
int z2 = FAST_INT_DIVIDE(z1, X2);
int x2 = z1 - z2*X2;
int x4 = FAST_INT_DIVIDE(z2, X3);
int x3 = z2 - x4*X3;
int x1odd = (x2 + x3 + x4 + oddBit) & 1;
int x1 = 2*x1h + x1odd;
int X = 2*sid + x1odd;

""")
    
    if sharedFloats > 0:
        str.append("#ifdef SPINOR_DOUBLE\n")
        str.append("#if (__CUDA_ARCH__ >= 200)\n")
        str.append("#define SHARED_STRIDE 16 // to avoid bank conflicts on Fermi\n")
        str.append("#else\n")
        str.append("#define SHARED_STRIDE  8 // to avoid bank conflicts on G80 and GT200\n")
        str.append("#endif\n")
        str.append("extern __shared__ spinorFloat sd_data[];\n")
        str.append("volatile spinorFloat *s = sd_data + SHARED_FLOATS_PER_THREAD*SHARED_STRIDE*(threadIdx.x/SHARED_STRIDE)\n")
        str.append("                                  + (threadIdx.x % SHARED_STRIDE);\n")
        str.append("#else\n")
        str.append("#if (__CUDA_ARCH__ >= 200)\n")
        str.append("#define SHARED_STRIDE 32 // to avoid bank conflicts on Fermi\n")
        str.append("#else\n")
        str.append("#define SHARED_STRIDE 16 // to avoid bank conflicts on G80 and GT200\n")
        str.append("#endif\n")
        str.append("extern __shared__ spinorFloat ss_data[];\n")
        str.append("volatile spinorFloat *s = ss_data + SHARED_FLOATS_PER_THREAD*SHARED_STRIDE*(threadIdx.x/SHARED_STRIDE)\n")
        str.append("                                  + (threadIdx.x % SHARED_STRIDE);\n")
        str.append("#endif\n\n")
    
    for s in range(0,4):
        for c in range(0,3):
            str.append(out1_re(s,c) + " = " + out1_im(s,c)+" = 0;\n")
    str.append("\n")
#NEW!
    for s in range(0,4):
        for c in range(0,3):
            str.append(out2_re(s,c) + " = " + out2_im(s,c)+" = 0;\n")
    str.append("\n")
    
    return ''.join(str)
# end def prolog

def twisted_rotate_with_swap_register(x):
    str = []
    str.append("// Re-use registers here!\n")
    x = x if not dagger else -x
    for h in [0, 1]:
      for c in range(0, 3):
            strRe = []
            strIm = []
            for s in range(0, 4):
                # identity
                re = id[4*h+s].real
                im = id[4*h+s].imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe.append(sign(re)+in_re(s,c))
                    strIm.append(sign(re)+in_im(s,c))
                elif re==0:
                    strRe.append(sign(-im)+in_im(s,c))
                    strIm.append(sign(im)+in_re(s,c))
                
                # sign(x)*i*mu*gamma_5
                re = igamma5[4*h+s].real
                im = igamma5[4*h+s].imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe.append(sign(re*x)+in_re(s,c) + "*mu")
                    strIm.append(sign(re*x)+in_im(s,c) + "*mu")
                elif re==0:
                    strRe.append(sign(-im*x)+in_im(s,c) + "*mu")
                    strIm.append(sign(im*x)+in_re(s,c) + "*mu")
            #use h2_re(0,c) and h2_im(0,c) as temporal regs 
            str.append("\t" + h2_re(h,c)+ " = "+''.join(strRe)+";\n")
            str.append("\t" + h2_im(h,c)+ " = "+''.join(strIm)+";\n")
      str.append("\n")

    for h in [2, 3]:
      for c in range(0, 3):
            strRe = []
            strIm = []
            for s in range(0, 4):                
                # sign(x)*i*mu*gamma_5
                re = igamma5[4*h+s].real
                im = igamma5[4*h+s].imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe.append(sign(re*x)+in_re(s,c) + "*mu")
                    strIm.append(sign(re*x)+in_im(s,c) + "*mu")
                elif re==0:
                    strRe.append(sign(-im*x)+in_im(s,c) + "*mu")
                    strIm.append(sign(im*x)+in_re(s,c) + "*mu")
            #use h2_re(0,c) and h2_im(0,c) as temporal regs 
            str.append("\t" + in_re(h,c)+ " += ("+''.join(strRe)+");\n")
            str.append("\t" + in_im(h,c)+ " += ("+''.join(strIm)+");\n")

            str.append("\t" + in_re((h-2),c)+ " = "+h2_re(h-2,c)+";\n")
            str.append("\t" + in_im((h-2),c)+ " = "+h2_im(h-2,c)+";\n")
      str.append("\n")
    
    return ''.join(str)+"\n"

# end def twisted_rotate_with_swap_register



def gen(dir):
    projIdx = dir if not dagger else dir + (1 - 2*(dir%2))
    projStr = projectorToStr(projectors[projIdx])
    def proj(i,j):
        return projectors[projIdx][4*i+j]
    
    # if row(i) = (j, c), then the i'th row of the projector can be represented
    # as a multiple of the j'th row: row(i) = c row(j)
    def row(i):
        assert i==2 or i==3
        if proj(i,0) == 0j:
            return (1, proj(i,1))
        if proj(i,1) == 0j:
            return (0, proj(i,0))
    
    str = []
    
    projName = "P"+`dir/2`+["-","+"][projIdx%2]
    str.append("// Projector "+projName+"\n")
    for l in projStr.splitlines():
        str.append("// "+l+"\n")
    str.append("\n")
    
    if dir == 0: str.append("int sp_idx = ((x1==X1m1) ? X-X1m1 : X+1) >> 1;\n")
    if dir == 1: str.append("int sp_idx = ((x1==0)    ? X+X1m1 : X-1) >> 1;\n")
    if dir == 2: str.append("int sp_idx = ((x2==X2m1) ? X-X2X1mX1 : X+X1) >> 1;\n")
    if dir == 3: str.append("int sp_idx = ((x2==0)    ? X+X2X1mX1 : X-X1) >> 1;\n")
    if dir == 4: str.append("int sp_idx = ((x3==X3m1) ? X-X3X2X1mX2X1 : X+X2X1) >> 1;\n")
    if dir == 5: str.append("int sp_idx = ((x3==0)    ? X+X3X2X1mX2X1 : X-X2X1) >> 1;\n")
    if dir == 6: str.append("int sp_idx = ((x4==X4m1) ? X-X4X3X2X1mX3X2X1 : X+X3X2X1) >> 1;\n")
    if dir == 7: str.append("int sp_idx = ((x4==0)    ? X+X4X3X2X1mX3X2X1 : X-X3X2X1) >> 1;\n")
    
    ga_idx = "sid" if dir % 2 == 0 else "sp_idx"
    str.append("int ga_idx = "+ga_idx+";\n\n")
    
    # scan the projector to determine which loads are required
    row_cnt = ([0,0,0,0])
    for h in range(0,4):
        for s in range(0,4):
            re = proj(h,s).real
            im = proj(h,s).imag
            if re != 0 or im != 0:
                row_cnt[h] += 1
    row_cnt[0] += row_cnt[1]
    row_cnt[2] += row_cnt[3]


    def twisted_rotate_half(x):
      str = []
      x = x if not dagger else -x
      if row_cnt[2] == 0: #projector defined on upper half only 
        for h in [0, 1]:
          for c in range(0, 3):
            strRe = []
            strIm = []
            for s in range(0, 4):
                # sign(x)*i*mu*gamma_5
                re = igamma5[4*h+s].real
                im = igamma5[4*h+s].imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe.append(sign(re*x)+in_re(s,c) + "*mu")
                    strIm.append(sign(re*x)+in_im(s,c) + "*mu")
                elif re==0:
                    strRe.append(sign(-im*x)+in_im(s,c) + "*mu")
                    strIm.append(sign(im*x)+in_re(s,c) + "*mu")
            str.append("\t" + in_re(h,c)+ " += ("+''.join(strRe)+");\n")
            str.append("\t" + in_im(h,c)+ " += ("+''.join(strIm)+");\n")
          str.append("\n")

      elif row_cnt[0] == 0: #projector defined on lower part only
        for h in [2, 3]:
          for c in range(0, 3):
            strRe = []
            strIm = []
            for s in range(0, 4):                
                # sign(x)*i*mu*gamma_5
                re = igamma5[4*h+s].real
                im = igamma5[4*h+s].imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe.append(sign(re*x)+in_re(s,c) + "*mu")
                    strIm.append(sign(re*x)+in_im(s,c) + "*mu")
                elif re==0:
                    strRe.append(sign(-im*x)+in_im(s,c) + "*mu")
                    strIm.append(sign(im*x)+in_re(s,c) + "*mu") 
            str.append("\t" + in_re(h,c)+ " += ("+''.join(strRe)+");\n")
            str.append("\t" + in_im(h,c)+ " += ("+''.join(strIm)+");\n")
          str.append("\n")
    
      return ''.join(str)+"\n"

    # end def twisted_rotate_half




    load_gauge = []
    load_gauge.append("// read gauge matrix from device memory\n")
    load_gauge.append("READ_GAUGE_MATRIX(GAUGE"+`dir%2`+"TEX, "+`dir`+");\n\n")

    reconstruct_gauge = []
    reconstruct_gauge.append("// reconstruct gauge matrix\n")
    reconstruct_gauge.append("RECONSTRUCT_GAUGE_MATRIX("+`dir`+");\n\n")

    load_spinor1 = []
    load_spinor1.append("// read the first flavr spinor from device memory\n")
    if row_cnt[0] == 0:
        load_spinor1.append("#ifndef DSLASH_XPAY\n")
        load_spinor1.append("\tREAD_SPINOR(FLAVORSPINORTEX1);\n")
        load_spinor1.append("#else\n")
        load_spinor1.append("\tREAD_SPINOR_DOWN(FLAVORSPINORTEX1);\n")
        load_spinor1.append("#endif\n\n") 
    elif row_cnt[2] == 0:
        load_spinor1.append("#ifndef DSLASH_XPAY\n")
        load_spinor1.append("\tREAD_SPINOR(FLAVORSPINORTEX1);\n")
        load_spinor1.append("#else\n")
        load_spinor1.append("\tREAD_SPINOR_UP(FLAVORSPINORTEX1);\n")
        load_spinor1.append("#endif\n\n") 
    else:
        load_spinor1.append("READ_SPINOR(FLAVORSPINORTEX1);\n\n")

    project1 = []
    project1.append("// project spinor into half spinors\n")
    for h in range(0, 2):
        for c in range(0, 3):
            strRe = []
            strIm = []
            for s in range(0, 4):
                re = proj(h,s).real
                im = proj(h,s).imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe.append(sign(re)+in_re(s,c))
                    strIm.append(sign(re)+in_im(s,c))
                elif re==0:
                    strRe.append(sign(-im)+in_im(s,c))
                    strIm.append(sign(im)+in_re(s,c))
            if row_cnt[0] == 0: #projector defined on lower half only
                for s in range(0, 4):
                    re = proj(h+2,s).real
                    im = proj(h+2,s).imag
                    if re==0 and im==0: ()
                    elif im==0:
                        strRe.append(sign(re)+in_re(s,c))
                        strIm.append(sign(re)+in_im(s,c))
                    elif re==0:
                        strRe.append(sign(-im)+in_im(s,c))
                        strIm.append(sign(im)+in_re(s,c))
                
            project1.append("spinorFloat "+h1_re(h,c)+ " = "+''.join(strRe)+";\n")
            project1.append("spinorFloat "+h1_im(h,c)+ " = "+''.join(strIm)+";\n")
        project1.append("\n")
    
    ident1 = []
    ident1.append("// identity gauge matrix\n")
    for m in range(0,3):
        for h in range(0,2):
            ident1.append("spinorFloat "+h2_re(h,m)+" = " + h1_re(h,m) + "; ")
            ident1.append("spinorFloat "+h2_im(h,m)+" = " + h1_im(h,m) + ";\n")
    ident1.append("\n")
    
    mult1 = []
    for m in range(0,3):
        mult1.append("// multiply row "+`m`+"\n")
        for h in range(0,2):
            re = ["spinorFloat "+h2_re(h,m)+" = 0;\n"]
            im = ["spinorFloat "+h2_im(h,m)+" = 0;\n"]
            for c in range(0,3):
                re.append("\t" + h2_re(h,m) + " += " + g_re(dir,m,c) + " * "+h1_re(h,c)+";\n")
                re.append("\t" + h2_re(h,m) + " -= " + g_im(dir,m,c) + " * "+h1_im(h,c)+";\n")
                im.append("\t" + h2_im(h,m) + " += " + g_re(dir,m,c) + " * "+h1_im(h,c)+";\n")
                im.append("\t" + h2_im(h,m) + " += " + g_im(dir,m,c) + " * "+h1_re(h,c)+";\n")
            mult1.append(''.join(re))
            mult1.append(''.join(im))
        mult1.append("\n")

    #now for the second spinor flavor:
    load_spinor2 = []
    load_spinor2.append("// read the second flavorr spinor from device memory\n")
    if row_cnt[0] == 0:
        load_spinor2.append("#ifndef DSLASH_XPAY\n")
        load_spinor2.append("\tREAD_SPINOR(FLAVORSPINORTEX2);\n")
        load_spinor2.append("#else\n")
        load_spinor2.append("\tREAD_SPINOR_DOWN(FLAVORSPINORTEX2);\n")
        load_spinor2.append("#endif\n\n") 
    elif row_cnt[2] == 0:
        load_spinor2.append("#ifndef DSLASH_XPAY\n")
        load_spinor2.append("\tREAD_SPINOR(FLAVORSPINORTEX2);\n")
        load_spinor2.append("#else\n")
        load_spinor2.append("\tREAD_SPINOR_UP(FLAVORSPINORTEX2);\n")
        load_spinor2.append("#endif\n\n") 
    else:
        load_spinor2.append("READ_SPINOR(FLAVORSPINORTEX2);\n\n")

    project2 = []
    project2.append("// project spinor into half spinors\n")
    for h in range(0, 2):
        for c in range(0, 3):
            strRe = []
            strIm = []
            for s in range(0, 4):
                re = proj(h,s).real
                im = proj(h,s).imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe.append(sign(re)+in_re(s,c))
                    strIm.append(sign(re)+in_im(s,c))
                elif re==0:
                    strRe.append(sign(-im)+in_im(s,c))
                    strIm.append(sign(im)+in_re(s,c))
            if row_cnt[0] == 0: #projector defined on lower half only
                for s in range(0, 4):
                    re = proj(h+2,s).real
                    im = proj(h+2,s).imag
                    if re==0 and im==0: ()
                    elif im==0:
                        strRe.append(sign(re)+in_re(s,c))
                        strIm.append(sign(re)+in_im(s,c))
                    elif re==0:
                        strRe.append(sign(-im)+in_im(s,c))
                        strIm.append(sign(im)+in_re(s,c))
                
            project2.append("spinorFloat "+h1_re(h,c)+ " = "+''.join(strRe)+";\n")
            project2.append("spinorFloat "+h1_im(h,c)+ " = "+''.join(strIm)+";\n")
        project2.append("\n")

    ident2 = []
    ident2.append("// identity gauge matrix\n")
    for m in range(0,3):
        for h in range(0,2):
            ident2.append("spinorFloat "+h2_re(h,m)+" = " + h1_re(h,m) + "; ")
            ident2.append("spinorFloat "+h2_im(h,m)+" = " + h1_im(h,m) + ";\n")
    ident2.append("\n")
    
    mult2 = []
    for m in range(0,3):
        mult2.append("// multiply row "+`m`+"\n")
        for h in range(0,2):
            re = ["spinorFloat "+h2_re(h,m)+" = 0;\n"]
            im = ["spinorFloat "+h2_im(h,m)+" = 0;\n"]
            for c in range(0,3):
                re.append("\t" + h2_re(h,m) + " += " + g_re(dir,m,c) + " * "+h1_re(h,c)+";\n")
                re.append("\t" + h2_re(h,m) + " -= " + g_im(dir,m,c) + " * "+h1_im(h,c)+";\n")
                im.append("\t" + h2_im(h,m) + " += " + g_re(dir,m,c) + " * "+h1_im(h,c)+";\n")
                im.append("\t" + h2_im(h,m) + " += " + g_im(dir,m,c) + " * "+h1_re(h,c)+";\n")
            mult2.append(''.join(re))
            mult2.append(''.join(im))
        mult2.append("\n")

#Additional stuff:
     
    reconstruct1_branch = []
    #
    reconstruct1_branch.append("#ifndef DSLASH_XPAY\n")     
    for m in range(0,3):

        for h in range(0,2):
            h_out = h
            if row_cnt[0] == 0: # projector defined on lower half only
                h_out = h+2
            reconstruct1_branch.append(out1_re(h_out, m) + " += " + h2_re(h,m) + ";\n")
            reconstruct1_branch.append(out1_im(h_out, m) + " += " + h2_im(h,m) + ";\n")
    
        for s in range(2,4):
            (h,c) = row(s)
            re = c.real
            im = c.imag
            if im == 0 and re == 0:
                ()
            elif im == 0:
                reconstruct1_branch.append(out1_re(s, m) + " " + sign(re) + "= " + h2_re(h,m) + ";\n")
                reconstruct1_branch.append(out1_im(s, m) + " " + sign(re) + "= " + h2_im(h,m) + ";\n")
            elif re == 0:
                reconstruct1_branch.append(out1_re(s, m) + " " + sign(-im) + "= " + h2_im(h,m) + ";\n")
                reconstruct1_branch.append(out1_im(s, m) + " " + sign(+im) + "= " + h2_re(h,m) + ";\n")
        
        reconstruct1_branch.append("\n")

    reconstruct1_branch.append("\n#else\n")#for next flavor
    reconstruct1_branch.append("//check this!!!\n")
    for m in range(0,3):

        for h in range(0,2):
            h_out = h
            if row_cnt[0] == 0: # projector defined on lower half only
                h_out = h+2
            reconstruct1_branch.append(out2_re(h_out, m) + " += " + h2_re(h,m) + ";\n")
            reconstruct1_branch.append(out2_im(h_out, m) + " += " + h2_im(h,m) + ";\n")
    
        for s in range(2,4):
            (h,c) = row(s)
            re = c.real
            im = c.imag
            if im == 0 and re == 0:
                ()
            elif im == 0:
                reconstruct1_branch.append(out2_re(s, m) + " " + sign(re) + "= " + h2_re(h,m) + ";\n")
                reconstruct1_branch.append(out2_im(s, m) + " " + sign(re) + "= " + h2_im(h,m) + ";\n")
            elif re == 0:
                reconstruct1_branch.append(out2_re(s, m) + " " + sign(-im) + "= " + h2_im(h,m) + ";\n")
                reconstruct1_branch.append(out2_im(s, m) + " " + sign(+im) + "= " + h2_re(h,m) + ";\n")
        
        reconstruct1_branch.append("\n")

    project1_branch = []
    project1_branch.append("// project spinor into half spinors\n")
    for h in range(0, 2):
        for c in range(0, 3):
            strRe = []
            strIm = []
            for s in range(0, 4):
                re = proj(h,s).real
                im = proj(h,s).imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe.append(sign(re)+in_re(s,c))
                    strIm.append(sign(re)+in_im(s,c))
                elif re==0:
                    strRe.append(sign(-im)+in_im(s,c))
                    strIm.append(sign(im)+in_re(s,c))
            if row_cnt[0] == 0: #projector defined on lower half only
                for s in range(0, 4):
                    re = proj(h+2,s).real
                    im = proj(h+2,s).imag
                    if re==0 and im==0: ()
                    elif im==0:
                        strRe.append(sign(re)+in_re(s,c))
                        strIm.append(sign(re)+in_im(s,c))
                    elif re==0:
                        strRe.append(sign(-im)+in_im(s,c))
                        strIm.append(sign(im)+in_re(s,c))
                
            project1_branch.append(h1_re(h,c)+ " = "+''.join(strRe)+";\n")
            project1_branch.append(h1_im(h,c)+ " = "+''.join(strIm)+";\n")
        project1_branch.append("\n")

    ident1_branch = []
    ident1_branch.append("// identity gauge matrix\n")
    for m in range(0,3):
        for h in range(0,2):
            ident1_branch.append(h2_re(h,m)+" = " + h1_re(h,m) + "; ")
            ident1_branch.append(h2_im(h,m)+" = " + h1_im(h,m) + ";\n")
    ident1_branch.append("\n")

    mult1_branch = []
    for m in range(0,3):
        mult1_branch.append("// multiply row "+`m`+"\n")
        for h in range(0,2):
            re = [h2_re(h,m)+" = 0;\n"]
            im = [h2_im(h,m)+" = 0;\n"]
            for c in range(0,3):
                re.append("\t" + h2_re(h,m) + " += " + g_re(dir,m,c) + " * "+h1_re(h,c)+";\n")
                re.append("\t" + h2_re(h,m) + " -= " + g_im(dir,m,c) + " * "+h1_im(h,c)+";\n")
                im.append("\t" + h2_im(h,m) + " += " + g_re(dir,m,c) + " * "+h1_im(h,c)+";\n")
                im.append("\t" + h2_im(h,m) + " += " + g_im(dir,m,c) + " * "+h1_re(h,c)+";\n")
            mult1_branch.append(''.join(re))
            mult1_branch.append(''.join(im))
        mult1_branch.append("\n")

    reconstruct1_final = []
    reconstruct1_final.append("//Store the data:\n")#no gamma5
    for m in range(0,3):

        for h in range(0,2):
            h_out = h
            if row_cnt[0] == 0: # projector defined on lower half only
                h_out = h+2
            reconstruct1_final.append(out1_re(h_out, m) + " += " + h2_re(h,m) + ";\n")
            reconstruct1_final.append(out1_im(h_out, m) + " += " + h2_im(h,m) + ";\n")
    
        for s in range(2,4):
            (h,c) = row(s)
            re = c.real
            im = c.imag
            if im == 0 and re == 0:
                ()
            elif im == 0:
                reconstruct1_final.append(out1_re(s, m) + " " + sign(re) + "= " + h2_re(h,m) + ";\n")
                reconstruct1_final.append(out1_im(s, m) + " " + sign(re) + "= " + h2_im(h,m) + ";\n")
            elif re == 0:
                reconstruct1_final.append(out1_re(s, m) + " " + sign(-im) + "= " + h2_im(h,m) + ";\n")
                reconstruct1_final.append(out1_im(s, m) + " " + sign(+im) + "= " + h2_re(h,m) + ";\n")
        
        reconstruct1_final.append("\n")
    reconstruct1_final.append("\n#endif\n")

    reconstruct2_branch = []
    #
    reconstruct2_branch.append("#ifndef DSLASH_XPAY\n")     
    for m in range(0,3):

        for h in range(0,2):
            h_out = h
            if row_cnt[0] == 0: # projector defined on lower half only
                h_out = h+2
            reconstruct2_branch.append(out2_re(h_out, m) + " += " + h2_re(h,m) + ";\n")
            reconstruct2_branch.append(out2_im(h_out, m) + " += " + h2_im(h,m) + ";\n")
    
        for s in range(2,4):
            (h,c) = row(s)
            re = c.real
            im = c.imag
            if im == 0 and re == 0:
                ()
            elif im == 0:
                reconstruct2_branch.append(out2_re(s, m) + " " + sign(re) + "= " + h2_re(h,m) + ";\n")
                reconstruct2_branch.append(out2_im(s, m) + " " + sign(re) + "= " + h2_im(h,m) + ";\n")
            elif re == 0:
                reconstruct2_branch.append(out2_re(s, m) + " " + sign(-im) + "= " + h2_im(h,m) + ";\n")
                reconstruct2_branch.append(out2_im(s, m) + " " + sign(+im) + "= " + h2_re(h,m) + ";\n")
        
        reconstruct2_branch.append("\n")

    reconstruct2_branch.append("\n#else\n")#with next flavor
    reconstruct2_branch.append("//(check this!!!)\n")
    for m in range(0,3):

        for h in range(0,2):
            h_out = h
            if row_cnt[0] == 0: # projector defined on lower half only
                h_out = h+2
            reconstruct2_branch.append(out1_re(h_out, m) + " += " + h2_re(h,m) + ";\n")
            reconstruct2_branch.append(out1_im(h_out, m) + " += " + h2_im(h,m) + ";\n")
    
        for s in range(2,4):
            (h,c) = row(s)
            re = c.real
            im = c.imag
            if im == 0 and re == 0:
                ()
            elif im == 0:
                reconstruct2_branch.append(out1_re(s, m) + " " + sign(re) + "= " + h2_re(h,m) + ";\n")
                reconstruct2_branch.append(out1_im(s, m) + " " + sign(re) + "= " + h2_im(h,m) + ";\n")
            elif re == 0:
                reconstruct2_branch.append(out1_re(s, m) + " " + sign(-im) + "= " + h2_im(h,m) + ";\n")
                reconstruct2_branch.append(out1_im(s, m) + " " + sign(+im) + "= " + h2_re(h,m) + ";\n")
        
        reconstruct2_branch.append("\n")

    project2_branch = []
    project2_branch.append("// project spinor into half spinors\n")
    for h in range(0, 2):
        for c in range(0, 3):
            strRe = []
            strIm = []
            for s in range(0, 4):
                re = proj(h,s).real
                im = proj(h,s).imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe.append(sign(re)+in_re(s,c))
                    strIm.append(sign(re)+in_im(s,c))
                elif re==0:
                    strRe.append(sign(-im)+in_im(s,c))
                    strIm.append(sign(im)+in_re(s,c))
            if row_cnt[0] == 0: #projector defined on lower half only
                for s in range(0, 4):
                    re = proj(h+2,s).real
                    im = proj(h+2,s).imag
                    if re==0 and im==0: ()
                    elif im==0:
                        strRe.append(sign(re)+in_re(s,c))
                        strIm.append(sign(re)+in_im(s,c))
                    elif re==0:
                        strRe.append(sign(-im)+in_im(s,c))
                        strIm.append(sign(im)+in_re(s,c))
                
            project2_branch.append(h1_re(h,c)+ " = "+''.join(strRe)+";\n")
            project2_branch.append(h1_im(h,c)+ " = "+''.join(strIm)+";\n")
        project2_branch.append("\n")
    
    ident2_branch = []
    ident2_branch.append("// identity gauge matrix\n")
    for m in range(0,3):
        for h in range(0,2):
            ident2_branch.append(h2_re(h,m)+" = " + h1_re(h,m) + "; ")
            ident2_branch.append(h2_im(h,m)+" = " + h1_im(h,m) + ";\n")
    ident2_branch.append("\n")

    mult2_branch = []
    for m in range(0,3):
        mult2_branch.append("// multiply row "+`m`+"\n")
        for h in range(0,2):
            re = [h2_re(h,m)+" = 0;\n"]
            im = [h2_im(h,m)+" = 0;\n"]
            for c in range(0,3):
                re.append("\t" + h2_re(h,m) + " += " + g_re(dir,m,c) + " * "+h1_re(h,c)+";\n")
                re.append("\t" + h2_re(h,m) + " -= " + g_im(dir,m,c) + " * "+h1_im(h,c)+";\n")
                im.append("\t" + h2_im(h,m) + " += " + g_re(dir,m,c) + " * "+h1_im(h,c)+";\n")
                im.append("\t" + h2_im(h,m) + " += " + g_im(dir,m,c) + " * "+h1_re(h,c)+";\n")
            mult2_branch.append(''.join(re))
            mult2_branch.append(''.join(im))
        mult2_branch.append("\n")

    reconstruct2_final = []
    reconstruct2_final.append("//Store the data:\n")#with gamma5
    for m in range(0,3):

        for h in range(0,2):
            h_out = h
            if row_cnt[0] == 0: # projector defined on lower half only
                h_out = h+2
            reconstruct2_final.append(out2_re(h_out, m) + " += " + h2_re(h,m) + ";\n")
            reconstruct2_final.append(out2_im(h_out, m) + " += " + h2_im(h,m) + ";\n")
    
        for s in range(2,4):
            (h,c) = row(s)
            re = c.real
            im = c.imag
            if im == 0 and re == 0:
                ()
            elif im == 0:
                reconstruct2_final.append(out2_re(s, m) + " " + sign(re) + "= " + h2_re(h,m) + ";\n")
                reconstruct2_final.append(out2_im(s, m) + " " + sign(re) + "= " + h2_im(h,m) + ";\n")
            elif re == 0:
                reconstruct2_final.append(out2_re(s, m) + " " + sign(-im) + "= " + h2_im(h,m) + ";\n")
                reconstruct2_final.append(out2_im(s, m) + " " + sign(+im) + "= " + h2_re(h,m) + ";\n")
        
        reconstruct2_final.append("\n")
    reconstruct2_final.append("\n#endif\n")

    if dir >= 6:
        str.append("if (gauge_fixed && ga_idx < X4X3X2X1hmX3X2X1h) ")
        str.append(block("{" + '\t'.join(load_spinor1) + '\t'.join(project1) + '\t'.join(ident1) + '\t'.join(reconstruct1_branch) + 
                          ''.join(twisted_rotate_half(-1)) + '\t'.join(project1_branch) + '\t'.join(ident1_branch)  + '\t'.join(reconstruct1_final) + "}" +
                         "\n{" + '\t'.join(load_spinor2) + '\t'.join(project2) + '\t'.join(ident2) + '\t'.join(reconstruct2_branch) + 
                         ''.join(twisted_rotate_half(+1)) + '\t'.join(project2_branch) + '\t'.join(ident2_branch) + '\t'.join(reconstruct2_final) + "}"))
        str.append("else ")
        str.append(block(''.join(load_gauge)+ ''.join(reconstruct_gauge) + "{" + '\t'.join(load_spinor1) + 
                         '\t'.join(project1) + '\t'.join(mult1) + '\t'.join(reconstruct1_branch) + ''.join(twisted_rotate_half(-1)) + 
                         '\t'.join(project1_branch) + '\t'.join(mult1_branch) + '\t'.join(reconstruct1_final) + "}" + "\n{" + '\t'.join(load_spinor2) + 
                         '\t'.join(project2) + '\t'.join(mult2) + '\t'.join(reconstruct2_branch) +
                         ''.join(twisted_rotate_half(+1)) + '\t'.join(project2_branch) + '\t'.join(mult2_branch) + '\t'.join(reconstruct2_final) + "}"))
    else:
        str.append(''.join(load_gauge) + ''.join(reconstruct_gauge) + "{" + '\t'.join(load_spinor1) + 
                   '\t'.join(project1) + '\t'.join(mult1) + '\t'.join(reconstruct1_branch) + ''.join(twisted_rotate_with_swap_register(-1)) +
                   '\t'.join(project1_branch) + '\t'.join(mult1_branch) + '\t'.join(reconstruct1_final) + "}" + "\n{" + '\t'.join(load_spinor2) + 
                   '\t'.join(project2) + '\t'.join(mult2) + '\t'.join(reconstruct2_branch) + 
                   ''.join(twisted_rotate_with_swap_register(+1)) + '\t'.join(project2_branch) + '\t'.join(mult2_branch) + '\t'.join(reconstruct2_final) + "}")
    
    return block(''.join(str))+"\n"
# end def gen


def twisted_rotate_accum_dp(n, x):
    str = []
    x = x if not dagger else -x
    for h in range(0, 4):
        for c in range(0, 3):
            i = 3*h+c
            j = 3*(h+2)+c if h < 2 else 3*(h-2)+c
            strRe = []
            strIm = []
            for s in range(0, 4):
                # identity
                re = id[4*h+s].real
                im = id[4*h+s].imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe.append(sign(re)+"accum" + nthFloat2(2*i+0))
                    strIm.append(sign(re)+"accum" + nthFloat2(2*i+1))
                elif re==0:
                    strRe.append(sign(-im)+"accum" + nthFloat2(2*i+0))
                    strIm.append(sign(im)+"accum"  + nthFloat2(2*i+1))
                
                # sign(x)*i*mu*gamma_5
                re = igamma5[4*h+s].real
                im = igamma5[4*h+s].imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe.append(sign(re*x)+"accum" + nthFloat2(2*j+0) + "*mu")
                    strIm.append(sign(re*x)+"accum" + nthFloat2(2*j+1) + "*mu")
                elif re==0:
                    strRe.append(sign(-im*x)+"accum" + nthFloat2(2*j+0) + "*mu")
                    strIm.append(sign(im*x)+"accum" + nthFloat2(2*j+1) + "*mu")
            if n == 1:
                str.append(out1_re(h,c)+ " += ("+''.join(strRe)+");\n")
                str.append(out1_im(h,c)+ " += ("+''.join(strIm)+");\n")
            elif n == 2:
                str.append(out2_re(h,c)+ " += ("+''.join(strRe)+");\n")
                str.append(out2_im(h,c)+ " += ("+''.join(strIm)+");\n")
        str.append("\n")
    
    return ''.join(str)+"\n"
#end of def twisted_rotate_accum_dp

def twisted_rotate_accum_sp(n, x):
    str = []
    x = x if not dagger else -x
    for h in range(0, 4):
        for c in range(0, 3):
            i = 3*h+c
            j = 3*(h+2)+c if h < 2 else 3*(h-2)+c
            strRe = []
            strIm = []
            for s in range(0, 4):
                # identity
                re = id[4*h+s].real
                im = id[4*h+s].imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe.append(sign(re)+"accum" + nthFloat4(2*i+0))
                    strIm.append(sign(re)+"accum" + nthFloat4(2*i+1))
                elif re==0:
                    strRe.append(sign(-im)+"accum" + nthFloat4(2*i+0))
                    strIm.append(sign(im)+"accum" + nthFloat4(2*i+1))
                
                # sign(x)*i*mu*gamma_5
                re = igamma5[4*h+s].real
                im = igamma5[4*h+s].imag
                if re==0 and im==0: ()
                elif im==0:
                    strRe.append(sign(re*x)+"accum" + nthFloat4(2*j+0) + "*mu")
                    strIm.append(sign(re*x)+"accum" + nthFloat4(2*j+1) + "*mu")
                elif re==0:
                    strRe.append(sign(-im*x)+"accum" + nthFloat4(2*j+0) + "*mu")
                    strIm.append(sign(im*x)+"accum" + nthFloat4(2*j+1) + "*mu")
            if n == 1:
                str.append(out1_re(h,c)+ " += ("+''.join(strRe)+");\n")
                str.append(out1_im(h,c)+ " += ("+''.join(strIm)+");\n")
            elif n == 2:
                str.append(out2_re(h,c)+ " += ("+''.join(strRe)+");\n")
                str.append(out2_im(h,c)+ " += ("+''.join(strIm)+");\n")
        str.append("\n")
    
    return ''.join(str)+"\n"
#end of def twisted_rotate_accum_sp


def epilog():
    str = []
    str.append(
"""
#ifdef DSLASH_XPAY
{
    READ_ACCUM(ACCUMFLAVORTEX1)
""")
    str.append("#ifdef SPINOR_DOUBLE\n")

    str.append("// apply twisted mass rotation\n")
    str.append(''.join(twisted_rotate_accum_dp(1, +1)))


    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str.append("    "+out2_re(s,c) +" += epsilon*(-"+out2_re(s,c)+ "+accum" + nthFloat2(2*i+0)+");\n")
            str.append("    "+out2_im(s,c) +" += epsilon*(-"+out2_im(s,c)+ "+accum" + nthFloat2(2*i+1)+");\n")

    str.append("#else\n")

    str.append("// apply twisted mass rotation\n")
    str.append(''.join(twisted_rotate_accum_sp(1, +1)))


    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str.append("    "+out2_re(s,c) +" += epsilon*(-"+out2_re(s,c)+ "+accum" + nthFloat4(2*i+0)+");\n")
            str.append("    "+out2_im(s,c) +" += epsilon*(-"+out2_im(s,c)+ "+accum" + nthFloat4(2*i+1)+");\n")

    str.append("#endif // SPINOR_DOUBLE\n")
    str.append(
"""
}
""")

    str.append(
"""
{
    READ_ACCUM(ACCUMFLAVORTEX2)
""")
    str.append("#ifdef SPINOR_DOUBLE\n")

    str.append("// apply twisted mass rotation\n")
    str.append(''.join(twisted_rotate_accum_dp(2, +1)))


    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str.append("    "+out1_re(s,c) +" += epsilon*(-"+out1_re(s,c)+ "+accum" + nthFloat2(2*i+0)+");\n")
            str.append("    "+out1_im(s,c) +" += epsilon*(-"+out1_im(s,c)+ "+accum" + nthFloat2(2*i+1)+");\n")

    str.append("#else\n")

    str.append("// apply twisted mass rotation\n")
    str.append(''.join(twisted_rotate_accum_sp(2, +1)))


    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str.append("    "+out1_re(s,c) +" = epsilon*(-"+out1_re(s,c)+ "+accum" + nthFloat4(2*i+0)+");\n")
            str.append("    "+out1_im(s,c) +" = epsilon*(-"+out1_im(s,c)+ "+accum" + nthFloat4(2*i+1)+");\n")
    

    str.append("#endif // SPINOR_DOUBLE\n")
    str.append(
"""
}
""")
    
    str.append("#else // DSLASH\n\n")

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str.append("    "+out1_re(s,c) +" *= -delta;\n")
            str.append("    "+out1_im(s,c) +" *= -delta;\n")
        str.append("\n");
    str.append("\n\n");

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str.append("    "+out2_re(s,c) +" *= -delta;\n")
            str.append("    "+out2_im(s,c) +" *= -delta;\n")
        str.append("\n");


    str.append("#endif // DSLASH_XPAY\n\n")

    str.append(
"""
    // write spinor field back to device memory
    WRITE_FLAVOR_SPINOR();
""")
    

    str.append("// undefine to prevent warning when precision is changed\n")
    str.append("#undef spinorFloat\n")
    str.append("#undef SHARED_STRIDE\n\n")

    str.append("#undef A_re\n")
    str.append("#undef A_im\n\n")

    for m in range(0,3):
        for n in range(0,3):
            i = 3*m+n
            str.append("#undef "+g_re(0,m,n)+"\n")
            str.append("#undef "+g_im(0,m,n)+"\n")
    str.append("\n")

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            str.append("#undef "+in_re(s,c)+"\n")
            str.append("#undef "+in_im(s,c)+"\n")
    str.append("\n")

#excluded clover term here!

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            if 2*i < sharedFloats:
                str.append("#undef "+out1_re(s,c)+"\n")
                if 2*i+1 < sharedFloats:
                    str.append("#undef "+out1_im(s,c)+"\n")
    str.append("\n")

    for s in range(0,4):
        for c in range(0,3):
            i = 3*s+c
            if 2*i < sharedFloats:
                str.append("#undef "+out2_re(s,c)+"\n")
                if 2*i+1 < sharedFloats:
                    str.append("#undef "+out2_im(s,c)+"\n")
    str.append("\n")

    return ''.join(str)
# end def epilog


def generate_twisted():
    return prolog() + gen(0) + gen(1) + gen(2) + gen(3) + gen(4) + gen(5) + gen(6) + gen(7) + epilog()

# To fit 192 threads/SM (single precision) with 16K shared memory, set sharedFloats to 19 or smaller
sharedFloats = 16


twist = True

dagger = False
print sys.argv[0] + ": generating tm_dslash_ndeg_core.h";
f = open('../dslash_core/tm_dslash_ndeg_core.h', 'w')
f.write(generate_twisted())
f.close()

dagger = True
print sys.argv[0] + ": generating tm_dslash_dagger_ndeg_core.h";
f = open('../dslash_core/tm_dslash_dagger_ndeg_core.h', 'w')
f.write(generate_twisted())
f.close()

