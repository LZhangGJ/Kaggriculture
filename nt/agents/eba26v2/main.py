"""Adaptive replay controller for Kaggriculture.

A complete season route handles capital, labor, farming, and planned sales.
Runtime logic stays narrow: actor-local WEED repair, demand-aware SELL-slot
ranking, near-clone premium preemption with exact quantity repayment, and
terminal liquidation. When a near clone is detected, the controller searches
three turns ahead first, then falls back to two turns and one while repaying
exactly the shifted quantity on its original due turn.
"""
import base64
import copy
import json
import math
import zlib


_ACTIONS = json.loads(zlib.decompress(base64.b85decode('c-rk<O>bM*5&bV(b74}HEO)2aOe{pP3`s7L8bT1DDGC(nBJHl|f3IRu<l~!}GiPS*eWcv1Oj9J^{l4>Y=A6&}Ir-bqzy12lZzq5HeDd+~?(XDacJlY1|M}N{J-+ey^4Fh#`^PW;etiA;<o(s{>hZ7Ki*G*t_|xTws~;|}Pi7}?Z`LQXg?Rh!{c81b@Q1tA>fPhp>-Ve6`;*!0(c3?)u5Uh^%;wvVf4seU_v!8Z?i*(h5C5I*_2=sC_n$uQo-`ly?eoccb$9=ztsib~@85rUwS8;!Vt*j+R@c|Nr_Rl%Za*-5>h`b0LAknq_tWFx-+$VS9@h?a5JYn}KcO{YH!Sueb7KG=y7|g!|DKP3ebAcSlq-`zerx#b@mybDzFloca_<qcZ`wn}EAX&yhx=oHa5v5PeNFxTTaW+$aKGI%`a6*)zr7p};HWK+Lv?w-x*ffGe(3H~qh_Fy9i2tnMhsiNy1X-<9{T0o56WTMK4Sag?&gy-T=EIZLf^J{`@?OAYrH0!kD6Elvi-_4pI+o9e%D?!W0gUZ$Isv}D2>);)iBdM8-6-5UTCq&&Dmz)#s^`C5hT`|d<R@3nRSPTFLN$z-WjxW_m1_b+yj)s+Wu+u$Yc+8?G-=#^dj)P=%c{A0$+Pxh0JHIi#BjWqL1EOU$5TY|MczZ_WtVn>MvhLt#ucsXwTTl10Q`p&;Dq7x#^Lu^2bM^N4s%i22(Iw+U{(?-`sp)3H{B;P7i(C_7iG0Km0c9l;L$Bvk`g?Q=|=Ym>PJlEeA=u;=D~H&c4{A?GfJDw{9Dh0Y*(|HN-n7$!nl8z=1JRhB)A9c3;EN{b)77gC&qKJIC&G(#xCr$puq8a&;x{rPwjs;1bGQJw4W8?lvx*efblw%T<y--h%gr_Z;^qOB`Sw@7~ae`3Jdu)24>ZJ(?z6V`2CIYx+vo<ruvvxf)D9Wt@GB+uE~UUP|nT3CG3x!ER>yteF>RM^_7Xk`ZEme|h`o_?<O2@ztpPrYY$-iDE_u#{^mLyWbusGBS6c5lAjgn_O1g$x2V07O!_h?S&cC&Zv@qt95{P=b+E4t+s+?Gn}m55AMAipFcTQAj9ZeCaFWNl<3(yNiz4!tY|7nVJ_`tcIEknGNZ*`v=TQ}6J&F#EzZ6y1J5jx7Q(#0ms#-{(&ygM_?T1s*zc~s>tz1ZJFJnfS&5qw2cl)_bc~XP#_Xwqu_4pik%J<&aH-hQ`<ng6sa(yMmcuG}f>U^X5%0r3_79rE0@m_zq_6@hl41^PXV9fGXjBR;6W;{xN88YF*^Bm<G2t?YXQ+jH)@q76+6QASXEK#l!>+Z-Wvy>+9{-uYLjK6_gZ5}mzAKa;Y`mDe``gR)H>=y*A0MCj#>99m9x^VQV$h7dF1C&$aYri^1~P0YU2FM*K3Nt-uz1{Nqg@iaD%%yn=ksQS9;Z|Q%z>x-_6P2I@aYZxH3K=$jnK2dH`ronlTm<vwI_3>7LgmtUN=e=LU;(22~Eu-u-lYaIF6K~i=@}W(pm&cf{Z~;FE?qlfU1^VTP2cz^yFL|RCCHN$7diMr(oWKoz+m*$ht+s)7JpC<cwEf_BtAHN|Awf*qiyoq<OA3OPx_;nA}}pyj{wlDLF;Aq1t@9jR-i_#(vZOiVpoik(#uc;8?fVdij7HsQ&3Td&tv~g*Gr<SVJjVt^t@5@9*eqt=guP@~BgGASeuW^yUFyPWmLkV_7E3g|)=l`2&y8-uLC^*)nDOIFm(nvb-u=uPiTUY0!}7IXk27ZL42<MZ^(;AmH<_;)x7X&11`2DWgx=Q(|#Q`xu}Xx>`%y7sxL66ONm87%+@f1V}B;uhp5Z*m7w(Yr}QHn|Z|gaOQ?t9P0=)$*o+OQD7zXgw1!<n6I{nBjy0%UOLBwJ3zAe5NgJQpeX3!-T_3H`ogUsELyW%IW{c#6y|?g%{3@U$y^n*7B&SIWMx2_{4QnR{cv^t{f5nFL>lwji1eR_t-x5geC`LPP2vjP@Z0PD+_6>1(5e}GVaf|k+K#&YS|h$e2LyG@*$pWMh+%-1G-T_cv7Mn}+`tTKvUPaawSLVs!&HEIa|uBxeZ-!Y8)na5x-_#+Hl95*64XlOmrl+qRqh2uvZO>+ch4=E{j?dLT^A*jaNBRi>Cm{8c$Z7|d-p7U^366Ry9u(zR6P;jNw92t*!B`eu6uSldmmMDgLO;9A+zWvd75n>_66)as{*OEKWD5~`CNsesm<I<^K}ppkWtJMZiY9WYw^<Hlk$f9WDf$sp9|?BoK_tH)+1N|0L`4E2#qplKKTX9B3EjH6xFBZT;#k5ru&H+(4aSwXay>PBfSY57}Fz)&0307?75ywMnj~QBSz{>8@OWn;OMPkm{X~5CTnG>AS!6BQ-9VA$%9Y~Yi5)gj^RR3g!Zjgz{TNsQwtuTvXj<$^8~h0dNcvfMzy2yq6}}Irt1J)D=3i^JPjH(Ut1yivQ@SpcC9R)<6O%Z51I6VHsC1M;J0wWk1fpnXk+*8_U0ppoYu{~p`n%@R~pA9R=4!T(4e;+Y5;GrU5o8_Z(oS{K?A*5Mh5z{O`DMq`qYi++BmHY^MGz6*Ji(tAuyycQW0w4c13{pwHZ@2-Nz*PqHc3xT-*BcRO<pG$fXF0!;z`_VZHwrBUaExVc~3s&p)hOTAVh(w840X@<+cSXXapbjR3QY51cQ}m6t-*4FyofZ!XMF#`<poTLF}%08e)Dd`s=kh54q^MI3dJX%K*@DS`*3=m4&9aS%OJM&OX(L@~LbS`Gj!8T7tp3`eGqgG6qpRFRK}&!2}SXCdZo!%sel(#{gG;CWaV(FRp0k(V+pN|gZxmYx|xB4%jWX!jX!Q+g+Mz=&Cm!Bxz|l~GtEblBh^84C4M7=|Jc$VLW%0#^*nVefP$!CfHICiWtjHaJOMg-~`iDgj=4=Pvs0TqQjoT$eMB(XmvcjUs4DX(g!KK?s$=<v)5>Y;S5msHEpn7JW6<Le2C~lcpM754*!Bgc%4gxe^#y_9NmL)29kp!0oxfqGJ#Gm0o0Bdl%bCpnFcrpb5HUEL2%*vA4d=p~<zrwmrPk&JL?<vMmy-7;U-{+Qs2YfiydUqJ+|hp_o`T!7vzhp%)J@fp_Hq*4k!+5!#T{0e{yv-b+xMh;eN<!2=IuT6WsX>UA0!u1C%1pFAwHjuN5@4)ROPE-{EiUvi-nwbFnBeM}heV4*UB^;IGO76XQhCW}#owNp7OM#smtv?;c<k0rqOUaKH=h62jtmUQWq2$}~l=h2>A(X8V5Qqjmj7~Br~!f0ATzd##L?MzD6Iiqr0Zvq41swCly6~GmiubS5{5n~cQK&HMq_r%to-2YWUf=oj1!xUbB1;~WM_&>3#Go%};IYY~b0K3>OtN<H|DnbeBwi;Wh94<SZ2o;zzm3#z44CTS0Th__Nq$z1>W|^?t2T3=B1Ku16nsEm)on2#oL}JayaS<%|!9HPXyTCT>5Z4dO3BZr`gu4K8RRTCbAumYEm{c3jPD?(+i!P<si$69MIliROs%Oub{P3E3ta>&$%$cPhDbR%~lqqPF%HK?S34D}-B;(2>KqwHXbE)gI%T=2^J6Z^N9##a}BjB;x`6d|zm)qI%ch_y@>15&vS1h`92i5kl-B%_1eE+$<U%v7=5~g>pmD-U&wZ?S8iz&hZp7Vcx7<PhbGmje{vqd|Hn#nFUb(ups#e%#>#7pj)Q*3Ge2Z(d~WD`Bp-pD4}KKZoXY5)cvF})(rs7yI>vgFM&*3s1!@Yd4erIm$-gC`*C*`Rbblb_@W4BBQVO?!EIcSs)hEs(kh_aI9p*fgeZ3@h{EJx1}o1_jC+MzB;AK$e$-q~;KZUzN6o=3ksqu?s-6oWnVE3;uY}*O}*hvgyx_Jm;Hk>}wNNtPkjyzt+5GGq|U2Do|vgEpnWZTp}=@iQ?U2(v!sV;nY*3Jm-<1{@{|WjXz^wQEq6L3|plTIPEE7zoU72Q`*gxC)s|mTP=1%A*FwoGXmQS3WvlOd?TLW$kJ8Lf+lHV#%c5%CAhxBujtGr0rOCnU9kLhMO5ae<nJI3)Q7ZY${uSx6p)FI;xcHHlcX|{YL-gL{z;1Zz%Ey~fn`F7y|xhRHb{fD-RUTyB~zD51uLLOD_9A@1=1So<uqf7)>0-jNh;ULX+Eyjl(LkQ;+Z#Y@%*Sg`>aEMe0f#n8$$L}{mVK23kQNeoi=ByHuu!<NBB~uZkb+0=Y|xUk0l?B+W&&H#{2WR`TqW@YL?b*pKTt%^UbC+!0}^CQm;7<@}cYMP2H5)ZDosq24<My4we<i$VMn~UJ_nNu#~g=A50Cit-xpaie#+Z>&KV4{QKUk?2053_uBd7Zh5ecl;fb}3PX}ed5I7G<z7oAQ{<~0O7$>fcBN2K%Lpk=gBD03NhKUm5$u+y3MMaA8q$4<NXIB%*8w(WwsV2J9p(1Qu!$k}0kGnMR43zZQWA74iC(EKh!!o+aVjwlYt_OkR}{LSr$tgshqjo%YF9meR7eXGCI5mVHp~kv9pFV_79C2_DYkv$R*ov(14D3l;A2|qHtn;WU7guv_%6UG5(3!C1`|GQfXWzqctJY+uE8PYh3Cj}#e<f&RY0=y@t`hdJ;ZW|;b{~zrKHi;1XWOMO5(c%?%;UB-ag=aQdZYV204Z*xJiz%=sqfQkZ>W}R@;NYM?@@+YGA04x#4DTK}{joTsz6>^=%(``YtGsC{G1I#__aQikcnP>l~^*F>FT0fB=jS$|NCmoRVC-%g+X&Fv{>V_hcE5Cu0hB(@+2mRa^5CEePzQ0+W%pCJHBzd>F~t5m$~6@HhyeQ;KR4jfeq0#(Ur0@AFqR8u+?WA+m+?NR&NsPBU(>fAZ-vj(Z`7vxCJ-K)h#R^l0%CC;}j&fZ-<1212p~<>dzV<u2!%*N23l1+^F1>|<s<$GPWHXZlg1EBCY>-HJe?Go<Oz_eKdp4pNmXy~_GMB)@y-r5oOkoHv=Jkr#^?sl~M+1=K^vm7K8aU9i#vOdUc}X;1AwMThOow(B`SU<h-K!72%THvP>G$@tmWaaOf03jiW2$q<EHX5U79YsDGnDqM`TeI>>Q$tRF&rL;5Pd&)b+1-2+L8tD)@xGo@Q{p1(CF<hYVwW9bWBU5?4*C_<s01z6`l~c=`d#K<wr2nnmCJ0|rL=UL&ea7=2E#_h~YNBI>G4aC_idVk()~@X2D5>$yMLg`}>C5EwLxEcWWk}TlbD#y?&;oh5sEgUMUW7@3HgB=Ks&pq$T0_v-IlR*jQ@ClAI~ab9cA(6vm5DQ@YT9WuemfL{xNcSGju$}5!A`ZV!JJIVCCwMOa9alpp<;A@3M~*Rb4iL_Q%5fF%~L_hGztP!#wZQ}!U&`$Y2E<%kgBa9UD!4=l)yrOat!ErVoXYbBZI<5WVa1I3Y5IQH!q5bdG-)MQZQg2SW6k5he+^9@tk42Ar6DXqxwu}7YHo{Y(QmEBgBLru=5F+HiM|nKB#s?Vh<#kiW}S+yEBo-flt;-3qGR7j?geUzg*3!D^J$Y2pr;yF;7F@uqw&Q6w1gdolfmaGXM7*vMQ6uIC9uB>~gxSA|GBFkH^u4S!{|6o8Lns8Nx&#It)+y(isoOn4k94NTg7PNG*szdn%PGAr6gZE$+mH%7l3Wn27bNMZzAWxpyD&i!O5%6K2L2CMY|zP$p}`VcI4_9e`4MVd9Q{iUurH3grt4k@i4FIIwNq7HEMpJ?hJ|SumRLjH{Q0-zaYo8#J^68N$tK&3=144`PNU^W7-c-5BPhpf7T+8dHxRm@+St-avN+VxIV8D!{xDCi93>=5&B7IrtU0rqv(JAsHb=CucdeTC=F=B0{c<$$XT`B?N(_XmemNcmMo1f1^}sD6b#s_$RrKQOFdgtrYH%1UgFfRIF&rOjp;5%p3UhK>z1Cc5BBEbw_@6ee*$62Fx&LY0*<9C_^s4?EvojCdSs?yX8(Z!mbkRWL;-8wa<!SV9FIripDD0)Z|i?{a2!dILDGv+VzltD8lEm*x=dJVvfjyF|~W1;8x30EDr^NQL*3}a6GlHA=~V|g@l^T7lt_N-k2?q9gwz%p?oFUzxetPX`puAN({#-juR?K1U(s7_tKESCc@Dg;)<w2)iB0On5CGl#wpL3rqZktFDuK&<M=}*=e~`|spos@_8IC|GgO-gkMcJSU<EY1Zye5JOp{CnUev<_{Q|WNZTvM|D6QzSK~xl8cwLx$$OT6t+AybQ3b6w*UOZHdL*7apR00sSG%+R&fQ{Dy&XYwo-Z~n#P`XsHo#PPksYe+|lEz%bxcC%ub5WW~S$dF<Kv{0*fx2xiK!G7{a*r#(NnQ<6j<pq#&FKj80%N9i8FR)7f)cI%3OJmcR8y0dL5^NbzY=DHx$r$uWuMCjI|w!qp^z?pW?bbJp$xFyh#;CRR=&8;(87ECRNN#6eB1uy(|ENd3kvZmku*Yl5m|sP`3*q`WCw(nMm%VOSkHc+nvgpmGZ~PLh(k=HK`7kWK}eQizfrQrgaQUG%t#gE+Gl17b|G`chvcqoumnb@K}K4t)knbO8g)9f?uSGO^0KpK+9f@x1p$>*G3B)7gC1=RNjC)7aYmt=!=2@neQ>1KPc2VdI7o1>vHQ+&pc5;w#RiL$LZNb{btHRkeWp>QW8F}Sor4ADq?EEGJYFarQ_66hycLOOuYKUy;DR>M;f}1k2<Bw?bl>bikTq>bwOL891GKCd&N;;p8JR^!GK_h^Q9890RJ|%JFem<|+^8^LkwxM)T5S^WsXG0Di<^VSDKQYofJ(5<6Vyp+LdscNF~pDmp#tJu;pTP@0uZKPU`sL0t`ZS=RZEQeP$pd2TULyjlMd2j!(iGQ875o2yoypsfNX&xi7zi2he#ADk?^dJ+5EhxkQpiDkn2vvg3)M$Il1;})y|?+O%>kqDP^AY?!UYGE>r;97d2AXfAT8|QL@~N%uO5I$dI-_&!=z7X{#WshsD9sDtc0S`dCb*cE~(;?)hK&t>?ca%xgQw2zKX+^yF|~164QG>6GL<I*5&Sz)_y%c{Z>v9S%5}`OZOdo22WJw85Eb*OUN*C+DW$Q9Rl&P=zidLEz$)m?MYf+dGRu&Cxz&Y1Tj6!Q8r}ty4Lf@{%VSmWh>igBiw_$5jDv>lhi*g5~w>)U73N8ZykDWB-@5a*EZhfNGR24EPYB*s6+CH_<~z1(wn`NQQ4Se!HPlrHIoyeausdgd1Hvgy@SIJ--p;i4^4Cu0{O~pHGWFOO=_VHyLE7WBLh@{MMgRwIVpejSqZJpNMALwF2s~BLtL{s-QLqwLnUHin>A;dn|`PraP$nLsXm1n76-&T&;(Me#9Fc&v(qFoBMcO)foR85P(rSx}^*832F#lbkq=J&SJ9PbGnm#bq7JaHS;IJb9ZQQLJt4OhylMC2}5{G;Ud%Z3Kp3jf!1NzLhu=&4y}vmLCQQvI<5OLwez}=;wloCaY|#sj}evU6R9l5rQOQ}c7ErXxK-x(aN%ekJ~AN4R$+kj91-S7cTyw)q&w(|#gt{&)YfUbsG}pv^rPJ`S`LmRhJytLJO&6Qo|Rwrp|b$}R$%7~Rck{z3x%!AyBsvoE6H4d_D4j1LFSS)W;M+p(F7uyNM495)fue}W=JXUQ*s7Ki4ouaD1yXHknV4vJ!JSGZJ%)-I<grOMM{??G*ikUW?Ol7WZNqRmXXczNa|TVp54d;0BS*TP-v}&$1wI5#77eG7lPFQwO=W4wGR$+NQ!-sfo)I*j-q@}ij)jTEf%Q|-G*81l>!m8&XE@trCYwt2pkiNW9TKc;M&<k7tv2fWjNaWu9~0Nb!gM#yyhpCVI*}j*%00#B|l-fi{po*%($xLCoV{H;&LtOS|`-|)C~n6gL*WwpI*w6Ls2qljss|YDkp7%yRQ<23vC<+ezIm<S$}&=XQsY25SH8D3^7Gw;6>A`o*ikl!UGw3(B$MAw;?)&kbrv-xy=Xcb}*79zo$`Ohhi-w>K~bE=Jh?W;wB1;A$5b;iPB8Mfn)S{h#WEnu5R!eT|!>91y^#NB<s^C>YQi-<U`iz+H%3r94Q;d#6a?`_<i20u#Ir>W5MxO4U^bFsv~jaQ>z9m*o;tX^FXH8GDtE%Gs+I<7snnYNwLizksjv~t7AeGlKq1wM4~?tnTCGW+PU!RP7JcP=}IM!T0Rph79a|T40BUNA2USS!@!}AU+j{^21l6D4yRg4MGPBBxwX`&q{tuG_*Nn!npB;{NHQ;P-jejHpOPdsA<&l-bJV8fw<;bPU2_Qnphm?JuG2&^|C1=cByAC3{c<bHD^yBOC^ZISBx%+nA?>=x?7Bs<seD1>dKi^I0$GS)NBP`MNJdkIhlGL~ijnOB3Q%;Jfutz;xWLkoqpk!5V1=UcSOPi`)$HV*P6kKAP){)cjxppW*J35!qHVqqah6%2RUK4Ny%i(HavlE(r3G_=icFlws?XY>Tok_evlpcnxcusB&a^b)ir_|d0WyA8<+(0+6xmt!(HhKlG{}42U48H{{U{3(Vw3|c-o`gIA8R?7M4`v<#ft$3wD6|YF67urS)M8Ly=Vxvu3JcF@=*$rdB>n`=`L@G=FrRXM+>%^RnldH9bTOk;DBI)3lr4U(QN0O@>^7cGvtSBK6G3s6blPr2vFuAL9rZ#A)I4MgryCAjt3<!A^|h8jpS(Pdh8_WLxOOfbf`STFaz{?R6msgh&T(`=Pu78Oi)3gesTkR6oMw@Re1qY$~L%m?l`I_g}7iuhX>CGl1}CHArkZRj4ZX-3kFK{&d54OK^g~+6ur{rHL<HCB#%}gb_xd33ZM!a(E%g!{229z_984&q;6-n$~RS2DYv0gdTdkuM^&*M;RBQ|iQxnlCMGXNt$YP)E3C!tvZ94Vc~p!Z48*Wd$4Sviz}{7j3_kndMg&dSe#Z;oI(Eu>=7oPiG(c4O@S|?!XKR9%ol|Q(4n;=r>R4qL+0v~-Hx8hrXt*{9sJcq1B?5NINdysBWZ_g`^L3Ci^oqIQ-<wo5(7J?F@m8tsB3VhqdRrhi7g~`>oW+-NAOw6vrgT{I4~87T@tJH@7!fA}UyPb!E((wotdyB>9HsZnT7~c2wq0cQ9VAFcAC4915D4j+19M3YhinQFkSF53sG&w=tp#^TO!N%6VEx)b`hC5d(FHwn3~gYg+)M<>v(IzU8v4;+%q>!U#Gl;g(no)D&&p&Ri{axxM7zM1kZg``e5(2J>`+0?xHUz58yGIkmkga7!WM`bDP*3M@FEiOXhE@Du8=2#PH|Viv)Tf10xpdhF^r)WN^)A0dLW%9*DpEpBk<{JMbA%0l9CLaPcr4LG{A!g26#s^ZS&nysFv=K;UbNl1agCMXTYC|@?l8cGg$VT5HeLF;ar0-hw{qn322hk(ZiQB$#DyXN@MzXmXd@Qu+!QMTdM2>nU~{wVM|U);(>EN7A>2x26=g$#g<t<zay_D0$<^ooL@Amr<7?#Nz2N0a{O`6g&B2-P@S{{g*+@Wpc5Un-h~&O>YL@^e?UVC3;')).decode("utf-8"))
__version__ = "adaptive-preempt-3x2x1"

_PRICE_FLOOR = 1
_DEMAND_ALPHA = 0.25
_MARKET_PARAMS = {
    "WHEAT": (25, 10000, 400, "sqrt", 0.8, "log", 0.2),
    "CARROT": (35, 10000, 450, "hinge", 1.0, "sqrt", 0.7),
    "TOMATO": (60, 10000, 200, "hinge", 0.4, "sqrt", 0.6),
    "STRAWBERRY": (120, 10000, 100, "sqrt", 0.7, "linear", 1.6),
    "MELON": (250, 10000, 300, "log", 0.2, "sq", 3.6),
    "EGG": (50, 10000, 332, "hinge", 0.4, "log", 0.2),
    "MILK": (160, 10000, 122, "sqrt", 0.6, "linear", 1.6),
    "WOOL": (200, 10000, 105, "log", 0.2, "sq", 3.2),
    "FERTILIZER": (100, 10000, 200, "linear", 0.4, "linear", 0.4),
}
_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
_SELLABLE = tuple(_MARKET_PARAMS)
_LIQUIDATION_ORDER = (
    "CARROT", "EGG", "FERTILIZER", "MELON", "MILK",
    "STRAWBERRY", "TOMATO", "WHEAT", "WOOL",
)
_WEED_STATE = {0: {}, 1: {}}
_WEED_REPLAY_STEPS = 8
_SHIFT_STATE = {
    0: {"last_step": -1, "due_step": -1, "due": {}},
    1: {"last_step": -1, "due_step": -1, "due": {}},
}
_PREEMPT_ENABLED = True
_PREEMPT_FRACTION = 2.0
_PREEMPT_MAX_BATCH = 30
_PREEMPT_MAX_CLONE_DISTANCE = 6
_PREEMPT_MIN_PRICE_RATIO = 0.0
_PREEMPT_MIN_FUTURE_QUANTITY = 4
_PREEMPT_START = 120
_PREEMPT_STOP = 680
_PREMIUM = ("STRAWBERRY", "MELON", "MILK", "WOOL")


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _copy_action(action):
    action = copy.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in (action.get("hands") or [])],
        "market": [list(order) for order in (action.get("market") or [])],
    }


def _seat(obs):
    return 1 if int(_get(obs, "player", 0) or 0) == 1 else 0


def _farm(obs, seat):
    farms = list(_get(obs, "farms", []) or [])
    return farms[seat] if seat < len(farms) else {}


def _align_hands(action, obs):
    action = _copy_action(action)
    expected = len(_get(_farm(obs, _seat(obs)), "hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(order or ["PASS"]) for order in hands[:expected]]
    return action


def _shed_access(size):
    half = size // 2
    return {
        (half - 1, half - 1), (half, half - 1),
        (half - 1, half), (half, half),
    }


def _projected_shed(obs, action):
    farm = _farm(obs, _seat(obs))
    private = _get(obs, "private", {}) or {}
    projected = {
        key: max(0, int(value or 0))
        for key, value in dict(_get(private, "shed", {}) or {}).items()
    }
    inventories = list(_get(private, "inventories", []) or [])
    positions = [_get(farm, "farmer", [0, 0]), *list(_get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    tiles = list(_get(farm, "tiles", []) or [])
    access = _shed_access(len(tiles) or 10)
    for index, unit_action in enumerate(unit_actions):
        if index >= len(positions) or index >= len(inventories):
            continue
        position = positions[index]
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            continue
        x, y = int(position[0]), int(position[1])
        if (x, y) not in access or not (0 <= y < len(tiles) and 0 <= x < len(tiles[y])):
            continue
        inventory = {key: max(0, int(value or 0)) for key, value in dict(inventories[index] or {}).items()}
        if unit_action and unit_action[0] == "DROP":
            deposits = inventory.items()
        elif unit_action and unit_action[0] == "PLACE" and len(unit_action) >= 2:
            item = unit_action[1]
            tile = tiles[y][x]
            structure = {"COW": "PASTURE", "SHEEP": "PASTURE", "GOOSE": "COOP"}.get(item)
            if structure and isinstance(tile, dict) and tile.get("kind") == structure and not tile.get("animal"):
                continue
            try:
                requested = int(unit_action[2]) if len(unit_action) >= 3 else 1
            except (TypeError, ValueError):
                continue
            deposits = ((item, min(max(0, requested), inventory.get(item, 0))),)
        else:
            continue
        for item, quantity in deposits:
            room = max(0, 100 - sum(projected.values()))
            amount = min(max(0, int(quantity or 0)), room)
            if amount:
                projected[item] = projected.get(item, 0) + amount
    return projected


def _public_signature(farm):
    keys = (
        "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
        "COW", "SHEEP", "GOOSE", "PASTURE", "COOP", "WEED",
    )
    counts = {key: 0 for key in keys}
    for row in (_get(farm, "tiles", []) or []):
        for tile in row if isinstance(row, list) else [row]:
            if not isinstance(tile, dict):
                continue
            for field in ("crop", "animal", "kind"):
                value = str(tile.get(field, "")).upper()
                if value in counts:
                    counts[value] += 1
                    break
    return (
        len(_get(farm, "hands", []) or []),
        len(_get(farm, "unlocked_quadrants", []) or []),
        tuple(counts[key] for key in sorted(counts)),
    )


def _clone_distance(obs):
    farms = list(_get(obs, "farms", []) or [])
    if len(farms) < 2:
        return 10**9
    left, right = _public_signature(farms[0]), _public_signature(farms[1])
    return (
        abs(left[0] - right[0])
        + 3 * abs(left[1] - right[1])
        + sum(abs(a - b) for a, b in zip(left[2], right[2]))
    )


def _shift_state(obs, step):
    seat = _seat(obs)
    state = _SHIFT_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "due_step": -1, "due": {}}
        _SHIFT_STATE[seat] = state
    state["last_step"] = step
    return state


def _repay_shift(obs, action, step):
    if not _PREEMPT_ENABLED:
        return action
    state = _shift_state(obs, step)
    if int(state.get("due_step", -1)) != step:
        if int(state.get("due_step", -1)) < step:
            state["due_step"], state["due"] = -1, {}
        return action
    due = {item: max(0, int(quantity)) for item, quantity in dict(state.get("due") or {}).items()}
    market = []
    for raw in action.get("market", []) or []:
        order = list(raw)
        if len(order) >= 3 and order[0] == "SELL" and due.get(order[1], 0) > 0:
            item = order[1]
            requested = max(0, int(order[2]))
            reduction = min(requested, due[item])
            requested -= reduction
            due[item] -= reduction
            if requested <= 0:
                continue
            order[2] = requested
        market.append(order)
    action["market"] = market
    state["due_step"], state["due"] = -1, {}
    return action


def _future_sells_at(step, horizon):
    if step + horizon >= len(_ACTIONS):
        return {}
    result = {}
    for raw in (_ACTIONS[step + horizon].get("market") or []):
        if len(raw) >= 3 and raw[0] == "SELL" and raw[1] in _PREMIUM:
            result[raw[1]] = result.get(raw[1], 0) + max(0, int(raw[2]))
    return result


def _preempt_shift(obs, action, step):
    if not _PREEMPT_ENABLED or not (_PREEMPT_START <= step < _PREEMPT_STOP):
        return action
    state = _shift_state(obs, step)
    if state.get("due") or _clone_distance(obs) > _PREEMPT_MAX_CLONE_DISTANCE:
        return action
    market = list(action.get("market") or [])
    if len(market) >= 10:
        return action
    remaining = _projected_shed(obs, action)
    for raw in market:
        if len(raw) >= 3 and raw[0] == "SELL":
            item = raw[1]
            remaining[item] = max(0, int(remaining.get(item, 0) or 0) - max(0, int(raw[2])))
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}

    # Near-clone routes often expose their next premium sale through public
    # state. Prefer a three-turn horizon against public-route competition; fall back to
    # two and one turn when the longer shift is not safe.
    for horizon in (3, 2, 1):
        future = _future_sells_at(step, horizon)
        if not future:
            continue
        shifted = {}
        trial_market = list(market)
        trial_remaining = dict(remaining)
        for item in _PREMIUM:
            future_quantity = max(0, int(future.get(item, 0) or 0))
            if future_quantity < _PREEMPT_MIN_FUTURE_QUANTITY:
                continue
            base_price = float(_MARKET_PARAMS[item][0])
            current_price = float(_get(prices, item, 0) or 0)
            # At the $1 floor, SELL does not add market inventory, so moving
            # that unit earlier cannot create queue pressure on the clone.
            if current_price <= _PRICE_FLOOR:
                continue
            if current_price < base_price * _PREEMPT_MIN_PRICE_RATIO:
                continue
            target = min(
                max(0, int(trial_remaining.get(item, 0) or 0)),
                future_quantity,
                _PREEMPT_MAX_BATCH,
                max(1, int(round(future_quantity * _PREEMPT_FRACTION))),
            )
            if target <= 0 or len(trial_market) >= 10:
                continue
            trial_market.append(["SELL", item, target])
            trial_remaining[item] = max(0, int(trial_remaining.get(item, 0) or 0) - target)
            shifted[item] = target
        if shifted:
            action["market"] = trial_market[:10]
            state["due_step"] = step + horizon
            state["due"] = shifted
            return action
    return action

def _tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (_get(farm, "tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError):
        return "LOCKED"


def _trace_actor_action(step, actor):
    trace = _ACTIONS[min(max(int(step), 0), len(_ACTIONS) - 1)] or {}
    if actor == "farmer":
        return list(trace.get("farmer") or ["PASS"])
    hands = trace.get("hands", []) or []
    return list(hands[actor] if actor < len(hands) else ["PASS"])


def _weed_repair_action(obs, action, step):
    action = _align_hands(action, obs)
    seat = _seat(obs)
    game = _WEED_STATE[seat]
    if step == 0 or step < game.get("last_step", -1):
        game = {"last_step": step, "active": {}}
        _WEED_STATE[seat] = game
    game["last_step"] = step
    farm = _farm(obs, seat)
    positions = [_get(farm, "farmer"), *list(_get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    active = game["active"]

    for actor, transaction in list(active.items()):
        index = 0 if actor == "farmer" else int(actor) + 1
        if index >= len(unit_actions):
            active.pop(actor, None)
            continue
        age = step - transaction["start"]
        if age == 1:
            unit_actions[index] = list(transaction["intended"])
        elif 2 <= age <= 1 + _WEED_REPLAY_STEPS:
            unit_actions[index] = _trace_actor_action(step - 1, actor)
        else:
            active.pop(actor, None)

    for index, (position, intended) in enumerate(zip(positions, unit_actions)):
        actor = "farmer" if index == 0 else index - 1
        if actor in active or not isinstance(intended, list) or not intended:
            continue
        if intended[0] not in ("BUILD_PASTURE", "PLANT"):
            continue
        tile = _tile_at(farm, position)
        if not isinstance(tile, dict) or tile.get("kind") != "WEED":
            continue
        active[actor] = {"start": step, "intended": list(intended)}
        unit_actions[index] = ["DIG"]

    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _align_hands(action, obs)


def _shape(name, value, threshold=None):
    value = max(0.0, float(value))
    if name == "hinge":
        if threshold is None or float(threshold) <= 0.0:
            return value
        ratio = value / float(threshold)
        return ratio + 8.0 * max(0.0, ratio - 1.0) ** 2
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    if name == "log":
        return math.log1p(value)
    if name == "log10":
        return math.log10(1.0 + value)
    raise ValueError(name)


def _market_price(item, inventory):
    base, equilibrium, scale, below_func, below_target, above_func, above_target = _MARKET_PARAMS[item]
    if inventory < equilibrium:
        amplitude = below_target * base / _shape(below_func, scale, scale)
        price = base + amplitude * _shape(below_func, equilibrium - inventory, scale)
    else:
        amplitude = above_target * base / _shape(above_func, scale, scale)
        price = base - amplitude * _shape(above_func, inventory - equilibrium, scale)
    return max(_PRICE_FLOOR, int(round(price)))


def _is_sell(order):
    return (
        isinstance(order, (list, tuple))
        and len(order) >= 3
        and order[0] == "SELL"
        and order[1] in _MARKET_PARAMS
    )


def _impact_score(obs, order):
    if not _is_sell(order):
        return float("-inf")
    item = str(order[1])
    try:
        quantity = max(0, int(order[2]))
    except (TypeError, ValueError):
        return 0.0
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    prices = _get(market, "prices", {}) or {}
    current_inventory = int(_get(inventory, item, 10000) or 0)
    current_quote = float(_get(prices, item, _market_price(item, current_inventory)) or 0)
    later_quote = float(_market_price(item, current_inventory + quantity))
    return float(quantity) * max(0.0, current_quote - later_quote)


def _demand_per_day(obs, configuration, item):
    town = _get(obs, "town", {}) or {}
    shops = list(_get(town, "unlocked_shops", []) or [])
    turns_per_day = int(_get(configuration, "turnsPerDay", 24) or 24)
    shop_interval = max(1, int(_get(configuration, "townShopSellInterval", 4) or 4))
    demand = 0.0
    for shop in shops:
        products = _SHOP_PRODUCTS.get(shop, ())
        if item in products:
            demand += (turns_per_day / shop_interval) * (2 if len(products) == 1 else 1)
    if item != "FERTILIZER":
        center_interval = max(1, int(_get(configuration, "townCenterSellInterval", 24) or 24))
        demand += turns_per_day / center_interval
    return demand


def _order_score(obs, configuration, order):
    score = _impact_score(obs, order)
    if score <= 0 or not _is_sell(order):
        return score
    item = str(order[1])
    quantity = max(0, int(order[2]))
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    current_inventory = int(_get(inventory, item, 10000) or 0)
    demand = max(0.25, _demand_per_day(obs, configuration, item))
    excess = max(0.0, current_inventory + quantity - 10000)
    urgency = min(1.0, (excess / demand) / 10.0)
    return score * (1.0 + _DEMAND_ALPHA * urgency)


def _rank_sell_slots(obs, action, configuration):
    action = _copy_action(action)
    market = list(action.get("market") or [])
    rows = [
        (_order_score(obs, configuration, order), -index, list(order))
        for index, order in enumerate(market)
        if _is_sell(order)
    ]
    if len(rows) < 2:
        return action
    rows.sort(reverse=True)
    ranked = iter(row[2] for row in rows)
    action["market"] = [next(ranked) if _is_sell(order) else order for order in market]
    return action


def _terminal_liquidation(obs, action, step):
    if step < 716:
        return action
    action = _copy_action(action)
    shed = _get(_get(obs, "private", {}) or {}, "shed", {}) or {}
    planned = {item: 0 for item in _SELLABLE}
    for order in action.get("market", []):
        if _is_sell(order):
            planned[str(order[1])] += max(0, int(order[2]))
    for item in _LIQUIDATION_ORDER:
        available = max(0, int(_get(shed, item, 0) or 0))
        extra = available if step >= 718 else max(0, available - planned[item])
        if extra and len(action["market"]) < 10:
            action["market"].append(["SELL", item, extra])
    return action


def agent(obs):
    try:
        step = min(max(0, int(_get(obs, "step", 0) or 0)), len(_ACTIONS) - 1)
        action = _weed_repair_action(obs, _copy_action(_ACTIONS[step]), step)
        action = _repay_shift(obs, action, step)
        action = _rank_sell_slots(obs, action, None)
        action = _preempt_shift(obs, action, step)
        action = _terminal_liquidation(obs, action, step)
        return _align_hands(action, obs)
    except Exception:
        farm = _farm(obs, _seat(obs))
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
            "market": [],
        }


def _kaggle_submission_entrypoint(obs):
    return agent(obs)


# ===================== TUNED PREEMPTION (v6) =====================
_PREEMPT_FRACTION = 2.0
_PREEMPT_MAX_BATCH = 30
_PREEMPT_MIN_FUTURE_QUANTITY = 4
_PREEMPT_START = 120
_PREEMPT_MAX_CLONE_DISTANCE = 6
_PREMIUM = ('STRAWBERRY', 'MELON', 'MILK', 'WOOL')
_PREEMPT_HORIZONS = (6, 5, 4, 3, 2, 1)

_V6_BASE_PREEMPT = _preempt_shift


def _preempt_shift(obs, action, step):
    """Same borrow-and-repay logic, but over a configurable horizon ladder."""
    if not _PREEMPT_ENABLED or not (_PREEMPT_START <= step < _PREEMPT_STOP):
        return action
    state = _shift_state(obs, step)
    if state.get("due") or _clone_distance(obs) > _PREEMPT_MAX_CLONE_DISTANCE:
        return action
    market = list(action.get("market") or [])
    if len(market) >= 10:
        return action
    remaining = _projected_shed(obs, action)
    for raw in market:
        if len(raw) >= 3 and raw[0] == "SELL":
            item = raw[1]
            remaining[item] = max(0, int(remaining.get(item, 0) or 0) - max(0, int(raw[2])))
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}

    for horizon in _PREEMPT_HORIZONS:
        future = _future_sells_at(step, horizon)
        if not future:
            continue
        shifted = {}
        trial_market = list(market)
        trial_remaining = dict(remaining)
        for item in _PREMIUM:
            future_quantity = max(0, int(future.get(item, 0) or 0))
            if future_quantity < _PREEMPT_MIN_FUTURE_QUANTITY:
                continue
            base_price = float(_MARKET_PARAMS[item][0])
            current_price = float(_get(prices, item, 0) or 0)
            if current_price <= _PRICE_FLOOR:
                continue
            if current_price < base_price * _PREEMPT_MIN_PRICE_RATIO:
                continue
            target = min(
                max(0, int(trial_remaining.get(item, 0) or 0)),
                future_quantity,
                _PREEMPT_MAX_BATCH,
                max(1, int(round(future_quantity * _PREEMPT_FRACTION))),
            )
            if target <= 0 or len(trial_market) >= 10:
                continue
            trial_market.append(["SELL", item, target])
            trial_remaining[item] = max(0, int(trial_remaining.get(item, 0) or 0) - target)
            shifted[item] = target
        if shifted:
            action["market"] = trial_market[:10]
            state["due_step"] = step + horizon
            state["due"] = shifted
            return action
    return action


def _v6_entrypoint(obs):
    return agent(obs)
# --- Current-GOLD public Replay rule imitation (generated) ---
_CGR_STREAMS = {
    '10C-4S-75L': json.loads(zlib.decompress(base64.b85decode('c-rk<O>Z1oa{Mnm_krE)=7(<_sn-&gGZZLl8|wiv7{F^7FxH2$Z^r(2YsLPks?3at%=e1o40vlbTUGD-WkyCu{``Mu|MBZ@fB)NWXaDl^*$-bn-o5|j)9vTSFAv+Zr^VTS{`%kl{y$&;^7Z52fBo&h{`S9LKmUC8;p4ZzY9D_1^4DK(fBN|2?cLep?A`tDY_T$5fBvxDej5DY^LG3G>({#v+uO&p#nt5NAGde+Kb<X>$De<CxPSlU-Q($hTs}ShV=?U5r;qRc{N>Z>P0K;wem>i6KR<qJ>reL&j~~8$I({|zFdm4{+q=8dTUXP!9v(M%mD7;nYfqo1Q{gpW^15*L+#ViV@@-C*qrUEc1>W`P?)Kd_8c(l3hyUQbZPIS=*8P7Oj;CqIx9@&BErwBF_cImzjE?a3cKZI)^0<B6K1>(U^t<uuahL9Lx`;mCf0-^KyEy;!&z&*)X3{$}m3DB(13Vd}Q~%!G?w97_N6$NR(Dl?@o`%c5^e_s;U(xAu_MbO7&Q35Z7`$aa_Rg3MhQrL5{f$0j`*DXuH+t@L=beVoPLsMWoxycBLwK~}vt`r;ZDhSehfm(2rTSRP-|%^KhH!tvICI42O&`SLJB}Z|p1q&F54nLmjeE<3_rHXb-uL-%!n<_f@P7wy>iS&z;X6Eba;sbf)+9Mh6BkIIr%un-%=Ufp7EJ9C@Ka+(^l8Dn`@6gCyT@Ptw0(H|c=z$&#%DsO!7INcu|&%6m}w3UxAvet;T}3VB9k8nSNY~CVFAAC^`DsEX&={h@0QwsjW&s6-Wl_8V1zphSFq<4XVu*key{dRJ7p&0J`8)y`bY=RaqJTYDRNcZPwatcEWA(jfyg5e?MHQ;IR0qd<e~!=52|GQDjNv;=6wEvr_<*8D!>zcoa-$ooCn9aKRwbKgM9N_-~`x~ynU(1QBxJ*W-n}5zdf!0)8u>Kv7y!~$Xz!KVq3u;&SMyTIf2E$o_Y`W9#SJ6hv=%64#|rBaD40Fz{=m9V%u9fml{G3S#P=%;QiWUY`}}QGA!7R38Bc0)5O}JFq@#|0ZfL1Jq8#33H77cC_t~sU~tGVbMQ`~tUnHL_4e4e$Nml<s{>eVrcMxWpZ;VV!l#haCJmqjA$j|5<G~_#PQzD<USk4JX$wFyi>LyE<WQ+h`$1H_)saOPd@wd2ZXf;<b*$fwpFj)5D7H6L$G#MYXgC%v6azarHU@bl6Ep!Sc+ig>dwy$f&^@E-I4HxB;&6zAuZ))6X^kER<rMLh2mSO!@2VNTZ(yM77|fKT0q-;C4V4J9eXylTy_pSfk8K5sHq3I`|L*$RcDKgV#wR{5Gd9-Vi23~ZaJ&0q`|$8r$C4=_OteGc+g%#wsyo~i8gvKZ=hF<((+eU_sf#jUGfr`*>cttu2vwA6E_-TdiVT6J9Grv^l&1A@{BXR#`QtQD?Qs1LcBs#E?HJc=az&;iP+SF3zf)g7H?wN~d7qveT8XFHD-E#Q1Y&mPaTQE(HtyY96zhA%#4a;H(WcRexp*oZpF(hq*a*NX@s3x;bB2yNLtZM^3W6pEm*DRH{_a~2jBDua|G1x^ujk|Yv$A1#a^C&kCcc)A&V@A9h!4?;pfVlm=vJT|^=Bo$mX8x5B5-ctcq34KoV29ghC*qIkkg?B!_@m)!kn78V7l+pMuoDWP6~@jP_6k?1~<1+GDNP44n{hSz2=(eL5c%{)HsWvc^4!)Z>QTk`lhEvWgDPj_CD!@;+#4bEuQAd)aYG~PckuY=dz+Mlc}-G923b6!mYXG<)){m%`uT5r8&BCENeod$N?f=s;-k-8nnT=e$jQNqZNuj%?2p=+TP^B4>6>mrVFCUcGGitE_TjBknU}mei!q)6i5L9WXCX40_}6=|CXdrBbN9v(gf0%Niu1R^?-H|%qG>7$5Fy+dIh@wq0s;ispJb1e6*+_zh!|Bf%cnSFTHFt8(*HTEcXK+X=G0vd*K)Wp{<b`XWM1;+^~N+lLvy_po1pL?4;aQmbUUrG#Va^TRi8K7C2*GINLJg9KhDj@K+95XFzpgXMXrm{{G>SN=qtuH6#tl+1=kE@w*R_yG;8aWy;?T@is0pYB)n0p|uIzpQSvHLav5$yS5o@of7+wSJn^>j!b{Yr~}R%)H)K)e8@t@_Nz}xswQ<LTm$F2JWlYz;QjIA-Jka>FY=S@FL_#D3rN1-($DW(iqL_CtPkTg>+BRei!Kc-VDPrZ>2qFoT#3$>7nE#L$vDX{l#plIQF}_s2|t<XLu1$gCDxk9&`2eF2e>f9ql7a|#YCvoSYB|z)Vgh+-j%o({ll}(IdkUPqDJ-{_jG`))|xd+i!oQrkwupb-Yc<%&iS@EoY{G&_Eq?|nsXvKzJ(9f>~iKBg}zy@)`4vRXD{c`>g_=*-np=>W7T(f+-ZT(3Y|#c6n0)__5~KwnY!XRPr1Qa)PSeF``9MBb~>@aVcycJR^FJ4)c0<Wlvg%eK(Or93NH6LIzGVfH0`N5d~wIdV6Hge;^xAbx@S&(Q)IG;xsN;xlAHH#hLCSk%5G~aDX=BI5;^U56`$f&_D6b>q|VII6_lT;Uu%CQISGRjrUPAtR>0xd;yPt&c$bVvK|yw)<C_lQ0$uNYqcUExoEhS6eNFKr5eRa~WyT418sd4ML^Q;jxE0bKd9f6x2XRcGvr*?gIj4>Zl6ynz?95o@8a=huUDk%xRnE4iaDiJUFv3wrshb^>3cO@QQk;ww;25|RibBLoF_H_SXfNSfD3dwUhsJ4qfmYQ|o!2mkjtTY}n)NnVZ`*So7(#aN%f7Eu;pElC$ZgwTj9iFM=X_nS8@j!#gv;etqM%2c)H4JEiuqQKg|G4!YCw6(*UYiLp{rEks5yj27)Qb(IPW3NS^zFnA~wAjO0rLLMIGz<BK#V*7C}$7xy0`4@GWBev^Dz*KN=k?ezT*6Ng>*sgkwwcP7z_()3^l7Er7<kDFwV*Z!3}w<d3zRH0ZQ<dHz&j#3B&S{>Xt5PG7l!bib+B9M*>6PxnYd$ih}pfr#3=P-$p(F{1@S1zG(8M>w1~+EQR1;8;%1(}9IHERxu|Moub6r^nJIK1zrLZGsu3fHwys!I^%@C^py`(lv`m0ZaOUE{-0@x~-Fw9J16_*b_E3%#3PxSe(RRxx<^Mj`~e7LCM8v!)l4%9NMUIO}#zVEm~mtAdip*tGY;ScH;G0dI3ah!SMIJHpgksd<~U4RD&2Kw6#3yS^j~ylLqW%ke+;$4XLIikA&sQsR9@q`MzlbmMN)wx72?QUA;~W1}9D`)C*2Q4cYL!0GnTDv^s1!&1DaFpP^RinNwzhTh!S5vQl1abf)UpWVq^UIWCC!!v-GZ^_8k!NQtC28k}dnVB2CsP^^#u9GC9c0XD1}OBvnOElE$&AZ!H^di#R#ui-`gIKo0Qm>B{|<y0uLY1sfy0!&nSjVlJ(#BdLXfR<dDV3~jekkAk|h!cx~%ZtX@5K;#bY0vh{U$P8|a)~NvaYb5SgzgJqXW9Wp(j<5YJ~FDlLry155&hz52<Tn2VwkpmqeX?f$5WZA8YFblXibHM57H}TmC0knN^2UvhsHd8hoN{CeLQTh+`%RxY5hbIhdI}iB6qW)RZ|coF;6p<4VaO!mj`T4;7p62Lp4_<#$1kH4%>50Ae#9x7>AIc#xg?)xMqdwxgm9)iio`^vJU6TR1s(&=N)w?7@tb=w+o=i_-WTQZk0|#D@eyeoX937*6?UrNt77MIjEH^<$!U<`$(1M4ZQSnE)$w|ngfG)4%E+UqrNM;+KJSWgdq*~yf`#yZ&)%zV~!0XUA6eVLMCQH_YcVS4!j-E6}MhoOUAv=Dv2YNex~9in~n~e8Wlu5BrU*-hJ3OLB<W*q4WNn(i239bNR-A;j@h2@1oF)DPDG$_StP2VN~74V*Vjet$LTreWb_ekop<yF6r(WGNK>Q^MkoqT%}YTXE-1(7yUt$1QITMo_wEttVcO9y6Yj;#B65XPoU&mAo3tRlx?e6K<9FfPzZk*7n(nUIAd&cEK%(H{Tf?JDBFJhiw922W*$tlEC9@}LAxFo&FFN<sqpD0sBPxW{9iWqyuc)bq95sOn-9e27*NO-}eYGWH>?CgjaR_zq&%@(vk`e6$CUQos(n>`J{3b~9k_xqFL`KG!O+kc#Xc$OJDH4}7$vsBP0|)JM@(i@C>!(7Mc6joOp-Qy-H3t?Q3J_s|gbtX#RCcx{2@VoHM~fj$7!hC&!KT)7;^1Py3%%ObB$vv%1ghVi_p4WeKV{nkeJK0EY8iHDD0HKNXrd$q*uxK8B~lHF^m{b0qB~emD?n4P)J)(@r^^t7ag-LXzHWLnw>2RnC-m92B5IU?o7YDPz-buX&EF9c@L_64kltV#O*Kay$Bx<gZiJ)3h2ABj^zDTfnvxibKD=K2p;>eRIO~u;u8&p_yk5xU43wCVu9Km_PAW%WSt_%F(1ruy+`^^DZ&Jg2`P7n(lJnA1yVl9f(5Ur^0t#R@Bvt=los4Iq!Yi;tej1MP(}h%y8u%60x2rhd6(WkHIn!2XTFv<p6)G!Jjr2O)g+!PKl{7(EE{_SrKp;^BI99b2UCMDv%dK3bLT$Qkj>t3Cs9Z`ZQKFO6z^Yki9DL|4$Ml*UT=pzgyj%x<z*Pq7!Gyd-ZbUk_K;la{fERMvUM`#7kh72H&FhNYUUl)xUNm%@tGU&X_&!4rw^(s6N5EF7D{N?ery1G?jjU`ZQdg(H6HPCi&ECk#wyR}a4e(iFBjI)Hx%o)tWbp3B0Fb=46Po=}o#__#sArEr$x<c|$3~&4m=jlFuo7m_4}qN6);-`veDov(3VNk08tPhI(1zE9pGc&sYD6lcbz;<28ow@~SV`J2H_(Li-Ohgot0N;^NuscjK;3n9ND>Aq4+uX>9r&w8J~U1YLu!q21XyVFqS;r`9AX&#N1kEUqrYivJG3Mr5>n<*-$e1U#l+>^^zIu7O?v#O;ZVH)_^+8hE?2$V?vurzi}MZy*Dj;$ELwEDE1SW@m{$qjw(H(VvWh2{-%h3Z)A1XB1<$@XryU^`4Tr6C=;|IdW6Q`3nHbbYO%SnYQun-icPtXhyEDb#M=<Wv=j}WDV)d$EqB$fJo1*qudOWnv`vvP+`t3F?71+wnr=`A=nzr-A>+|Iyi`595q!*wwI7?{^YofB!pefdJkwA|ENX#AYRBeMTD91>a(#eZ?>>4{8S8?3-s4$K^OCQ%F^ap)PU_U1k&6nH-09W+!A#F+OD8)+?A1NvgNO56r0i`JtUN4~H0NUEv@ox&HdjMww5r$!7HkBaym<y($T?q6l0gYUzdolYob|6r)h8md%NOpiyl^q1<T#NV~ykt@T7JEv4QVatvo-~8*BI7_x2muP;JLp;;wn$D&5-UrD<cYf5o~YD_pcYJxNV}{n5s0kH{zo!$Q8|F`<C;30?eEH1aGVI4vH;6Zw^J6JG}DSOzaWO!>dU%bJr4?Q^{8T_f|Km;vcD@uFvU`VbziwE;AWSYrN`zO0-V>{qj()uN8jaw7x~qglla(^_pUN&u0(W;sW%bFv4Qua$f0@U_5`8^y%x46d!z&)W`$7igC_NtisR`h16~kN4=J78pQ_JRNS^obh_Im4<&3jI>#$`MGaV-a#N#n0!vaehb7lYRO=+1WnIJ-~#Z(2u8>jG}=nDYWg{UfkGgQ#7llT5w@WDig3g$66zel{6Lc?8PMx#P#RhzXPkjYHA6jBpt#o=R1qubd1F~W%D_B-Vv>ze}>7z7B(xZCW0Z<?=bYj`Q)Y+9X$@m|pS3`rW$V`ROn@GxL1lqQo7D?GAd%(-c-bqJ>wq_TAoF-58{WUlTKVzc2I2tg50V>gJjQp?Xsl1fRJ0<3P`DNARy4)%#tBY5>zR)MDPkg5s;7}V?01qJeZ=JRv~Y7%G~kIBkK7CIJe6o#V~^Oy2}RJMqgcp<=1vq<zKd_ynvDA&u-q~4CFS5peMgm1{RBQY(H9-8iHI_P{Qs+igfg&x4{AEY+`y)v~nVWHq-?39jkH~j&Y89U|{sb%({3J{vdnpZ2;()j_3;Dl;$qK+!knbuTaB~p8E!!)G^mHClxI9MZUo}AMIEwQ%TXs|}c2*1DCWJ%_(y#Gyx=#i@>V*jRU%*sCGc^T)>s(Yv{N4$XnMDoui+^<ujryt1eXiN??htI3K8)*YAJbZY<iU6oYVR_eB2x`q8_(O7$r=5u$bS%K43T8yTD2ZH3DOO02aSL#>O=o?{2vOUVyALNx>UKx9&kJWQmglug!A#}DyhSxlqR#h0e@++Y>!`Cqs85)7YBG!_0`HySo)MhW!iG{3kcxCD{Ak~1%kZH@PrXKra@*N~4OK*fNX1Yw{HQ1qvb-IJCcu~j1&(TDqvcaY_z<|niM**KN~Fx%^v1e=DwX}V;!{$dlV^Zf3mg*=<xp&<paidABMHc0c0#_+#InSv!W9v$SMJA$p$@br1u>qFs|^1|^6Ny8ut8nYk)*pUEi11d5n(|rbU1I>l6+YW7`$p*O=HSuFc0F%g=j;*S)svnYF`xGM_fRJ9*%?>O0rAK3t?9~)4+7{0)#VWJxfLE#557o9a+H@@);<L2?2@?wO<^{j@0Rlo(Rsij{hIalAaz#B_rU<^#8IfQiQ_{nz#%K2K8Z}HHa{IRnR3|Z>MFu6RuVxMT@Lyl$wSpIVeX>s`n>PreZUJB*t8CG~F^GC#z4taxyqt5giRIY@43B@(iD;RvUq8_1YzHsL1N3spUm+oJu_#YVCEPT3Hz#n((Kxr{*krTSuqnt?%2dVD3(;-}?%cdk|7Hc0e187!mWdAoJsF2F=KF00Q3CW*s0TdmEAT)jA6;6-rQ2(S#BqCl4k7+B_=>r8v^5b<TPPvMbI$T6}gs#ll4L&}ESWdIUs>6kM8g0I@)&i-!X#DNON_YmaK0JjmP*>DXcdTd?;qFA4T4!Rrv^bpxUzBciz&XS$3CM3Uf!OCfd{z3Ck5T;)p!D?zIqjr^Aiwt;bTj$@;Ld21zHA>%%1`HpZgbL>qd3&X7Dg6b5K<EGWyqBC0n>d<&A;zT+h71sx61WoL93H~?AD@6?g^@BmF<btZ2LI^^j$9Uv&)FBuBpF{8WkAz@+$I?!aEaX}xYKB`cvJh;cmI>RO@ueiC$(l<3Z2^K+jiQ*l`B_2D#gkj&;(p00HQx}{&&iDf1t(#WEy1)=O{kbDltEdjFOh`mFWXof{?@R5c#q;_!jG9{ne92sLG#Il7PdkPK{#bhjnS`#o=E9xq?Qm{u!C#Ox9Unnkjv7(!}lmDkpod1!0RhUZ|QY`KruOs7P%Z^-7uAKu&He=T+_-SRO=!sV&&-ulEFoa<P*;$K@j3%*46!0TaxZv3FUJT`Waklj*Fg9O-d06EOREYYW11JbS<hzxScf!55X*rs)SA_pc6UQTE(7!U{Udmg8281q%Fua$Z7T<r~`pf&h39lshexwo}W*&JOnV6rSBSk+SgP0;r>xC6r1juc~JnUdB^9qa9|jM{j(!;A@Ex({H)<`!y4+Nff=+1ukrl|EzWZqedx3#&O5?PW7=((4IOsY+0i)r$!jt^>R?c%*)EzJNnA;b?tl#2GGnXP3Ol}hmZHBq#5!VFd$TJws>TS)Ac7K+AqXNb&okjR2!~c#P`c0%90&yWEuWhBRK+=@Oe+|z%R-v#CbcxQn6jmXh3QzRoCfDYL6#nq)Jpsk7sI7}yowI`#8Tu6`dqzUs4j=PvC*drtT!<a5pva~%&smoVo$&XLZucpM^0CAj^gcPGoPi^hPa71pFk5M?}H12^#+?u_6ZN4rCqI?Qlx?`6)erA;^76*^t_h8f@p|#4dQ+ZOSKQkq!hVu;$*TCKak3*B$jWC!*E^NNDRM#WFQ$01hp9_y;v@zv-?UY(Ot>?bZ{uUfDuH+@+o17iRVt=z;wv5$XsK05te`oA%OJ@>1p~^P|V`>*Iu5X%d=_<1?nyvlCzY;@<fQ7&sbX8T|IOga2Hf>5UTb=Pz(&m0X42C6Y9Oa$b<r*@2zEa+Z$*AB@3CA9lO5DWLE+rqX#Mv6#N2s8Y=2-B_R1aIFwU3>P)GbBf$H@ulM*yY$U`+-v)wMSfVmnUs@v;iI%EY<|N!Ypo#+4Ng-ZZaEChX6=`^sSXbwb)&HmO|CO}FHf<g_DVj<Fm0f-U^NJK}a6+3keUpJE9e|Y~j}q`I=l;|+WU^X-1-Ki6k_7}@y>;z5l_`#jj!ML&VV<B3X;0vmTCJ-J-GpvSqM(n_S^#J3Rmj0q3aqG`%4!#DF$i_u5DB7>pp~?KQ5+{zS)+kvaFyDoR=hYRb*WW0<Z5L8T&q<9NEl+B!BCY7AnY_k>p{n7Hf`svq({ih=JA=K24X?bYT2OhWH;(gPvALq*HGIIUl8v_?PG3wH#LUe4)+#U#O0+Xwq4jk2{=k3k*0yiP7J?xK{@X*A>m9Evk$=;cgyCkN%PEv?NF(62Gq$Au1Iw?3LtZJ6>AlXPzX!Bb(}l47#P-;w1Z700TFB^fhmv1sATi4M8F*NS1VaR>YIN9yQ~aJcO|!ZFXuu?NuAJT5)WMj0iQfhy77?EO^C~_!=KMwE!0WpJ_a=qs4M&x@nT{)xKc<C=CErDYXx$mUlPd86}P7{(|%`vS1VBn(mh))kLDv-h@k>~ZQdva;V)J)04eZ!+S5MPU%~wytt>3s#kG%769vi=ON<-9!>&9`L%6v{?!e_P?Y37cHa0FwN#`&aR!PjcR(%t2hwW}pAiydlHf^ByhGBCFy;mA#xM64*iZ4<Gz)I=0ZulB-A(TpXJf%Aoo}zUHVYjsUwpCgIE*nAPGz=sJo)IwrL;j085s-`qb6z3UQh+uu>YfP`@zdG}6YKCj9JQ4w_h1*7t^jHf6&4E0*$mFVoM0knz~y?dt~0>!L_ldL4);@lLTB<UGyE>ZSywAlIJB;it(p=_QX~R-eOXD9grvaMay=Gp(<X@y95HyD_F-q&Jb|->FEf<jm&kA(rUx8`cO9TuZ}m1)!DQsEB<gLx2*oTzAV?~lr)zCSM}4i_W<oe#VvO~ne$A9;oU%L`%4%i-HM#0NK;AZ98OA6^bXB5%iPRAk0P>*sRhji=pHgOn(V4=2YK^xj;%S$esg17C3$BzU(QW8J+4~==zvMM~E*sfkZ1;*z8B3>31bLk##I>HP3|2LqKZsLvCAWeAOSqh``lnyRej<sm3Mn%&TY%XmRW=D_-PtcYs+A!!7gTCwM<+@2)#vCcGe<WG9+$SP5SAdsp?57cm_zQ?{?Kny&&A443<&R}m76qm%JOYQ*^!_Zs2XxpUi5kC{`0v?g67u2dq*odt`%K5O=q4=*-{m<Fav;BmLJ&E89=zU55O?B{=-Oy&vGj#7hcV=2DB=&aGIwiHwT9#1(Z7w$kW|4q*-5~Frs%!{5=j@ZWw4uCpHUb;te6zsRw-q9^jm*;!ObNje}oaD21sn(o8W8X!s-L)G|7I_xh%DX!rFmaw??MUUk#ydj%y=0Vuc9FTAT+RQ(VExDC9-XMY#<KH57kOv>3h>LN|$$hx%@6ZK!TT7aC_cHRpHh2a@LwKG$~LJ{jL0P=pBJ9A!pevU9(f-c`Gm6vMPzFU<CrJ#T$RkTdjp2c0rCdKKKR6!q@;atHWDVZ1NclvNAQtVDwY|9=UING2vDH8R(wA@Ts-gm1=s1b#w+Au<DF#Kc86N=lO8%2<c8X;)gl162&6fH+w#oSU>5QTGyEgG(fe5vw-lEzz;7^WiaDVIyNZR5G>Vztzk?znF1n#VODR89`uRJCLdv(>$Cck%$Zh5<pn9)MEGw5*=!rTfAdpq3IMQRL+0xdvH5Q_^+dlLLh{TWZ!ywZ$+}mzkMU6D<^N+-GhrEM4+<q@ETH_-ic}3w=7bK|3LjPAv*67w4%6iM5x4O=T<4&)HJ84OL-NT`!eRO&L>S*;k65h}GFCg%IUR<jOdRDL2Ux1bd!TSf!}XZmWmiYECaMgo@;yVYYkoA|wS)wN@y0F6PkdQ|+GNk}^a}52Y6yRZBchWCf=5ph_W1Z1eAshfGSZ?J{W<$uvw!Ol@rU#>M(d`iHIckyC|1C4U;ie$-;ERlQC-QzUf-3E|SboLYt;>;g0wA%#{K8A^c_Vrz$QHw%Ba75GP$wGR^a;qp*;O-+2n9!Ntn6PKh*f7|;5cMpSMQqp7OI|jWz&y)b*^*D1ho-F@Iti5?%@4crE512W~y0AiA0ztK~ZqG~N{&q`RmJ%w{kVwFlS6**%rAvov6SANjcxy}sJZ-EjaA7Az$6aFmqs#A;M94s4Esex&h$v0OvIX@VFI=b7<sht|3i3o1!VJ9F`+3niep3B`KVA0NXkw^nc|oxTh%oaPS?>hGcuWO>r(-17fMfHbS`^eS2MRKDZjmmeUx3~cMsSubs@JBRI%LKW&qYxu>!FAP$HF}<og-y>z~?4St#gG=M0Z$=v&M)nYT7%HQ7I;llU#6B1FT8m2gMlWvx=z_sh~s3>0-17C^O3bvtW7GB!);e19)epaRY52Re^el??>gbaxy|DBuuuc>nk2wGJ@0&UruApjqP*a;U#gIST(99)}?9u1^7c0XYbyu)CY$n@EV%+{^9<U+$ti7K+8-(@uVopba_saOrq++PC9tq>rLFyle07F*Cya;sld799%ipCE)3K3evrhDM6Q^gSvg{_T84_kT8`q$Xi%~(!Mzwb;gBWDnnT|WPBjv?fQ;sr_rid%Ty&(h&+Zxr1wOf_0bc9evd*P7AjBGbrS@Le6!8qS$O@>v*6EwI9KZym-s&7sXhfZ5m>*!FEXvEi6DLc+nnH=Jm!n04;&h=JTTUlw&FwQ0!fPE7`(j=YfE1m{&;ASbbw9(nug7Jee?S>c8->xj6xPG>ouNaZd;m>#tDrpn<LUnaZv*U^')).decode('utf-8')) ,
    '6C-12S-100L': json.loads(zlib.decompress(base64.b85decode('c-rk<U2j`ia{MoP=7TAb`pBEc=I$7)85y!2Vlxm11MCI?g3ZGuZ^8ceII=`u-cwy&)#uQX2YRDv>fZBxx~r?JfBD~&fB)_GfBgOTlYjZ;<cH7iZ{Gd-;ripJ&v%=X`^CwB{Ptge`|mIR^X20|e*67D|NcK;KL2v^<NJsI)js_2`LDlT|NQ=^>zk9s$=loQ$>Oy6`s0tA&4<Z<eB5l_effI(<L3I)$>L)6^-r6d+n-Ms%iYgE+}*zW{PxrSFV64p|GAiU?8E!FfBF1j|EA@nZ@-*uHy=MewDsrPyH7tpeA<0A`*1iAA2&BQ`?sFX-}>~p$*Vv^rmx+9nok94!0dJ5?7<%HTJkU_%Y(i?{))Wo!_D>EO*EdUKTm%E-ZpDDdF$h!Ovkfn$HR9&?-#>CUms^G_*pu_o9p@e_siq@)8=lzi00oNt{%8_m-9vR@%Hn45w(l+PyfF&4!)W7j!k7dIEMo~8>M~!-d;a0&8HuI-I<fFTXVS|uJ)x*qcHtdI$dD@p~(R|p;^J?EstXl#%wYi&5X6b(P!*&-09FAJa@kH_CwfCle#X!;DXH%9<BUrIp~5mvgpvsC-14H`dG@}<nsuI@bQEJbCk`SK8U+_>^^)sdq1KN-oWk0z30J?zoe5s_W5+ehjifSZzpdW`rP!xGdy;7tDFVaWOA4q7s!~W&d*k7`+oBl%<U2K)5eS#(}K6RH#eKNpML$*=I+z`oA>{AcqR-Qyz)zoB~pIJk>=p()}FK{+(SD@WcK6WDqr6l7T}9s|Hk~z`?#umx2gTtX_El+t}!1cMmSiwf<31=t8h=?UOg^t%S`5dnD#d7V>*Dqu{R7-=BmI?*#p^Fpik)onMWYnkLo%B{%G9fq5~BVs$}~r8;JVm`TP@4r_c3OfT#3v&|5Z~2VmUqAK4m%`Q~qd6JlHD?Xw=2nyLgh`^JX#>(j<RO}_Vm4Yg81?z&+R+Y0UBd<dg2XR!E}Q}6B;AvMx*$gW!HkgV7bySGjbEdTBl+uqYTYX}jt-gPI?`?bs1pcidrShyV%LXnQsl(pY5o2cbMOooCzMi>1a^-Hl)f?g$qkwb>g!8?bt{y4zZ>tkOZ`yD=32eA4~ogm;o|H)wpKZTsuHh>a@<lA>Q9xQX`G<>D#H5zzIUjQ<*h$<k6he|o^CsFlYN0wdi!PxwGefO_X$Hv|G23jCSvC&ZN`cfRC=~(no4BEkIV~{&CK^Kt12YuJEuW$7R9T`=-L79$J4u=4I<zU(E*XYxr>?5A?pzoiEuA1rlCI-5W!OS@t^gaV`s6?3C2e&k-SF_>uvArPCI<uS}e|Pz?-CJX7jfsy-$HsazVm^MlyWak=xx4!-uw+UIQ`({M?U06fIvj2a4Z0`d=lv0&r*DWjSr?^YGtP0R>cttu2vwBnSoYM^6qy1`9-PDxl&<x$`>?ye^T%nR+SBzr*`b)}Ixw!;<%-NlptuU6ep_FEZD!T{L!Z7jv=UEkuQbAL6NuT)<0_co(YSZ7QLN{RiJdz@(Yn)!IlC_$pF(gfu@Q(>f{s_kbEb}YhP+g+6$VXAF2U{X?af0D3^eo}|MPKzzMPNmPRfSe-g(EpHNKXP&V@A9j1Mu0pfVro=vGKOif6@M%e#pX5jYq)UJFzokd`#sP$*3ib2_wOn0jAJm{S)QO!r;-s8DUFGlj(@sJ8i31~)fRGKAMefRWB)uU-=oq&Oi+9cB@9?*gOq?R2A~Z+==-wgDSv^hqBSaO$*Z@jOqaM(+YX$;`YxmKAkwrp7KkCQ@vEEpdQum0M*nb=5Z2MDCR41j})*8NtE>OHdjwNIMPM0Ip_s)#*rv;%jFE6nx#@^1%-|+@PK%Y_ehMZl0&@Sr5{kPE+$@&X)o=Fz{>_Vlt3GF#p#{YBgh!@1js7wK+*N?XfJ_4x*W+dSW?CX0=x!{DV&8cs7tJ+Y5}DeghN#qKyKBsfgQH1S7-PFzzX{Zz{$HunwAdY?JzKjd<Jn(9&6m|5svGvUju}N+aGbZZekS<t!^0puXM7Gr0}g{-1T;v0<fMV}DX~lv0qFK<@1SYdIY(_k`tE_a&0Zx!^Sr78uxint1J%fDxiODz@X%%16Axen*1_KN%4}y}$X(p-%-pE@URapRwTg?p;Urz0PthX?MROvrab!T}EEug+U4tum&UVqPeyeJ|_pH>iC3?%~xDa{psn!%(ehnAHn;w#nPIq2Yx-tgHX1Tz|gpE0A!#nggg$}yHymXWFM8%tJTG$xs@dZT%&Zfp8ejUXGRaE3-&>ituFyAZW8C>nqRS3WG1m)lRyV>hnN?HNuH*mM0|yIAW`dy?+gQFJycKyeqJQOnnq>7n^V9e1yd7Pvt($|HUOKc+YClQ3$8W?nmAo~EvdL-D1)v_!cwg`U-M1-=anRay*gd|?vR(9orkPt$j)6>Xg`ua#=5Y7l9VYUG;%lVUPC00rjYyy=6Hw`H9lW*$cNX)eD{s%4jLX3w%1fkw=7-2lxz1w3Rv=w{7w_C8YsBX+Gn~qO-Q_6j5FtW5Amo;^v_}iHNMc91PfqWL(QWsia-rc*#wCAZk3>j3i6yQ*i8C2@|y%$;Vai{g;yZ5uqz-4|HKI>2b{Q-k-gjWFmcBc9Sj;^E419o6L9b`niA2nH9DPi?uhfzCH^Is!BCH;Z46g}1rX=s2I2IGGS|nYA|WvrjAS8L+Os~`CgEmGf;D2`T=TeeVixgL`Km*_j@Z^iYq`)ikmCFF2I6c<fiR}gQpeGm!$<tFv34|RR&CUVv<jKc#~B*ltop%oY}ZIqGP$;8_?~ygJ!7kY#rhJO^}4VI>`omkIHsT99b`X-J70;8i$Q2TI!+w<1GwFH<uwmpwvsPPIV?f&z#<Y?eBBC#Ob?6e?s?QDL6*_W@MnQ=$@#-!#j*$iRb0&8CE-!AFJ7B<5ckX`FV%sfS)rXXO^0jx5;5%FDHqLMM3UiwBijI!K<Rb>{#`DOI{?43Z8!WK(%$6Nx<GR;j2l6tHb^8>um$sxy|~v2sBAQ;-%N4<qnv8o+DjDa&^v(Mj(0I5-4fWU)PwVw9l5+WXaHy_fu?GBmjuP;J<}l=s#BM2lpQl}S^+W^dh_~n@X(}xAZd3}3y-uB^p5`lx%Lq`@~|(bt3aS?8CQo{u8nl_v~_B-7R?#PU3hb^rx0;65rtGCUkgkaS#oG)@vd}8da?<ql5TqxzR)m7r8iTXQdd9ZVWN-*G)=S!Cpj8PRcHV$3qiNmD=a4^O%!m8j56dB9MI-S2>GXtddUcO{^?m29pRA4OF>Gaio%vtd~^!L9=)gO=3vKq4u5_<6}i^H$rJDs$<NP{u%{zc(HCm{=_-g)<Z=XGx3<!>r+SvJvh37U6-KsAyCEZYS5{gJg%?<M0W}p2X+1AACNDSYa}xTn#Psl_d7AaYTq@cS>SWjGx=f&}CHSzE2}*JYVj4r12*4eYE(;}{9C?9A)4*6{)ahUghZ8zF7Y=AKTj?=}Uz6X2gS|P<qH6c>z8_8f`q``G(mG+=(cY>qE5gl65<`^crPYoi*F1<gEu=mvbeMd-b0S>tf~xKy4E1P9Mqz2FjzwmiQt5tRE#{eKF;;AkIpO+oJXDaZOMr{v7oELYS^kvWL1*2-cFE~#N@Z7nX8aOBthNLmcnMGWw(H0PggQT1Ys3bD_JEHhmNx(h8j1C|sL=6YAAtWr^<W~5H=TjYZaLQ7(X1!C)M+OS;msQl(_7jDy3GtpVUZ-5YZb%LZ^d>|>?dO=P7{V6YCd+>N`2fJ2+A>s3cE@Pk5B`zRTWGyP&IbDzOPW_YiUl1=#L)c2n{fK29~I$stOo-g(PPKU@aVIkX1O_IC)LQkEie`8C!&_A|!e0Ob!u}LL=P{A0-G>LMl>XP@c;(jxGoWX##hsm!&gK%<;usLHsS>A#<c-C?>}QC=!P~D$9q(jKux1Hc%2#2!spIfR3v-iW<9S%Y<)eUHVWH!Q=(ol4ZouMDD$2t3Kwg`RgUFnuM08M$iEGv+R&7)Jd@}mvC6gU1G;@(CG4Fr|`WA&J+iH(OVo@qe<z%_7_Q)dd6~z872ow+ko77@oW+sBLIhQMl@meH-{cT#;_xE;B!n`pP^rcqV?Gt{DgROz1s9J@hDT|pi~DlXT*b|`pZQrGf@FAIx#IqBatLq*j}in?6j*ymAjzkNujb4&EI2?h)%NM+?!5vP?}<>j%#~K7HgteEa-N<vH@lnxNm=~?gu&8s%Xm-vN0oCXSH3i>0n$gik4=Q>=`F`So!HRNtS~c)Z{dZo3Tcr@(v8`LAG%!-Bp1UMPNnU?4^Oz{ACG$QaMzcF@+<<+k>vUQ9~vM!WA!E%37#ZYZ9A8+a^}7E)tDcl{@5#qjpJ<f-#`3P_Fu205i2{GKIZs90Z?`BH{{qRS-oj^WFPDcMu7Tw$x|@SB+O65ers-G|E81D+_@6EP21vQF3C6FgtW?Y2nt4HYGB;`%?%w%5I4+ZlMRm3v*t~)QQ89{SL8PX*a7)0wW}SZgE)N<ME$a#>Y_SM%T=>g*Ixn0a2k4D}$%D<EZs|I5oBU*ou;bwoDVXh%QwKRqwSj#o)br=2bSpeY!HDmm(;m)TCAcuU~Z1D^8Nkw@OfCgB~aiIq-KR=et-Qn`50U4=2YKHD3ir>zRvRc`?lyp&12u_o)z?;{IsGYLrOy0M?|EQIdG#o8$O5kvFr(e$PwyZ;&1JA_9~MdK6=L9SQI(IH~3&gZ)DdHkS;oMj(~XiQeD{<$Hw^tz*i%#+*Q4=Q-g;i8N^~r5)D&C2{O@j!K>f&t%B~^R5jGO%6zETS>}tida#5HtTmaJqmOhJc~g0&74*6Q7i69XC>Wwz82WIp~~OpfLIbAOc1j*E%a3;=n?K8+a`lhZ$ouJY!SufnZwnUGgs6)6tu2_ilCf23H3B()*X44QB-I*S@Tx9mRCof$TjADSUZb;+D7D7iVio=9Vw{~7Sg$N(i%_BmT1X+;+?0?{Q%wHfIq$Qa=Hu06X?y#Ee@O@7!r4u1}z6o)49L~=HVi|!6@@-h!ijC&84vD*u~4x5G+m_{+#je`=;ztz*R(mhaYd+LSz|qMxP*8UD4;$7!=}W5<tn3{Zo>bhGp|N#g`B#sIQY_(5{1L$1i>Gsg^5nqfUP|)qA|`0>{wYycqhvVNi3u#T2xuq<oWfHsG*(!Q#4RK$SV5?Ak=-U$?PIDfCY2KzDF{>}dDkrIq3ATD}x|(!}(gC>E1a)SZK@kO$oQok9*2hNQ0hK&CdTNyBubc)0s(mn`sEL->(_d>DFTNzqP5BFJ8ew9d~%K<@AvOzHXD^`2VfL}Pta^6lB?2Ie4)h0fLo<|^dtrnTb%jUSvYISkxdx?cy>tZdl$5@1@Gx=DC4dQ>1C6iy>If(X4GSw;y4EW*eYFE)k3Nl6Hgp2Y(Fz>?)}=PxobVR=;|<tkPI5oQdIm!i8D9n>0J+p)&xO-T|nxFz5A3r`D~^x$e<9Ge&KffQoYP5>gRMGwy7f*^US`YHexUilu8YXhTo?5=tawg=;R3Z^_vKR>Gy^}OWst<3cR5F2xdJAGRe^em$a2ZIeTvzOxs19MCfgPPDI!U9`!Qy3m{yJt8x4Af$&hP~?~Yj-7%vZ4U+?8(>{TG7Bx?;NOX0gp>neC#9sYt&JqY}bnLDV5Yxf1qAs?$oL5$Jif<eV&Ld`;;Y%meDTru)f{-yjO>|vL4FTGzfA_t0K0>I5Vm~p(?P?<jQJrG*~pHV#hR?9oy%*-F@!puF>ryT<C}EFVkr&6e_$hDmidbFm~KBS`DjRIH+QwRLLPOs2Y67k2f<D@2XFC)TLjgb!e;21`aZ*H36`a3}z9zPill;16VYY0-;VOPMwxU8tvdkFAk%!<NU-RG2u=Q+^e0X05w6Ah>XVWbb5e1&Rn%p_@l}3UK)<828uZo2Z&0`_`I8qCy6XCfndH@!zz1TG$VM$C>brt^{TNLLRv{qW?n?;csP~^r5M!{YZ=CNB2$qO;iy%xD{GKyMZ0zcip*98DD>j0x-Op}-uig3R5GEc!=}S9+{w$iM4*e8wHzm^Nf6Vt`bxNt(D?5tOA@<moWQ>^)y-F>v?eM*fo}V3*;RJo2th`Y@5DQlB(TS<jbo=cfj8$uDm=rs+;S<+i55TXSxUT=h^8~ovYZmnIF`)F#2<Yp=kmU!_|dlTaH$AoBX9^^Wc`W23jvd;^{2`%64%?!f2~R9t4j(lFYa8iw2X2-wkF+4{d`6(ksRtki^!&IsC6-<I#Q_1Pf~*W7PsFhMUeBZtVB1Ltw2e`(wjVOcw{sAERGDkaV~uEbYM%5fii_}O5n{}B=9iFQzmu9j7%-ps&EAafX&*8G)hHd#o!|6WR1pHd2tmY1U>=pG*;sX*>eDXsqmqVw|zy-PEP}H?I1boFu(wt_=>u9S2od@2H!w9D08R_N;P@Wll5`RnqY3kfF5pB8gA)QEoco{Y9idkea`%=YAqdkB|1rPW{H#*F;i9fNR%5N+kq=2>BQ!Vm}Z5E#1M%ZYgJEiU~$zF<s_4%)J_R;#f3$&346*W;Of30R$CH?vdS}POMPk3mlPpfMs%5m5jjDCUT}0uB0Bz6#Gbh?sNdS^*iNw$R+tY@)M=#;`0C~;pB4(2w#-5cRk(t{Do0kqF~ll~k;|uu3U7J%wWm_0ym~AsnNWF&NVJG~>KNuZ+TaHEr{+BH+bu7|%@)k1aq?0No!$yCMPlNKo`=g&(#c7-d-Epk7*K#f2JLBFeF+_6$Qc73rrvQh$<;c?YEeK<<bd8#`7(~?@U#r~5p_B$^S%)nXz_p*jihtq?iDr6Tl`(5P^zfqMX(c(2ZWtBu+}n!#@+-pNF7enVfPnUHt2YqG>M6p_{ravlm@$j@h4GL=6Lmo0INxHshL1uNzjL#W2-)d%t@C6nQVHgAHi?EIL6vDl%4?eFnqes`cjLuMe4_sWJL}vbGzulPt00OhyNlRDkaCHbap6U95HV&{Kn{QBAF-x4;9!gqBKPGE$6J@tRCTDSa;U=LRPRUc$-w98c_X{mJPR+)f9D<`F^Msjv|@AcA@TQnWcs3q#Uyz%6pz&I;a*_%AGXob1}YMuy`vul<E-iN@`ldVL8MA$f?G9;FNUE`>X_d+7;sGJ0~!2E{tv2n~X+J7<XB)U!$i1$Dychrot!$8qJG@^MS2$0ZE&r#(@qSa$KIq9v5h3J4;K`w###$=ANOODxO$+LbZ18L5#s~LbT@m`{XNqS9Q*KQ0Eznn<Y9ZdV1Nj)i;ZNAalcxlk1m7Er$zji_YR?XWOnjOZ(GBLKH`$Q`$&^Cd@oQU$aSB#s-5?B|$7YjAoKNNK83H<tEn?_PD>%<a!x}x!CQdYjV|S&YDDX5Lcucb!6ySyeR{>l(L(8N?lwHPfD~(OJXZdb@bvgg~Gg6QW`l=U`nQ9(lqJeY|doN;i_b)kyWazRHJ1|qX}YqkUYQ3nhY1!os@LifinM*_h%NRad_lGAzMI%`MTdIa8nXU$YR@M&ezTbW#<e*73s;oUq}1CS|n@oRQn!Is<v=ED+#!UQsG9WTzO94I0~$6P8kD6EQ)SS<XLUn@=KDYaq@Kagu5qNh8KO;ELGm3<er3Zn7{+D+C^Pdt5-rRw`P?IS7_gzMLmlOep|_?%O@}n;T(1D^imHRs;cfmpPMGlnT=vjFX7XxwpLxjPgAIlfAAE|@I*YTc#iWHuEH<jF4F&;+ios)s?;-(RBmIARjhASt7|1m(Pv8U5OCG<y^N40@H7gUq8;omj_A|^;9!b%jDaHah0NKWWy8r1nOvBjk7FeK7Uf=8jWr5C%E`0>P07l1N|SS2Sso$+5Bq4;JY;l_jd&c=x^?w12VsR6&=F{oNT}ATcIC7@NeL$(C5qZb%pfr#E)PDhob8Fkf}4{kQJsn^05bd5>HzZ<5Ia-k+oX|%7|2PZJ&JfV(f_vU9i>Z4<+s+ejud6=M5mVv#$i+`uRZBkD=FqUX3L&}+maNKJ!gY*kCLgBD5e<47bxSwY}aP7>a>f6B2zFXgV>I<SSeH%?<T74K(V8CH9sZP4Msk}HcS){<!wciIK7nASy`8n`%q07k%~t|hbfm!KRRNTKwa64Dt|Bt6YS>vJ-R3t4N!K8YhiwmbUzfHTXuEkxUyxAqWMucfwBa)Ap{Gx$}W$iRi+oh^e_zAhvhLUX7-n}lxC-w3|SK>(^Z<R0uI!pk|_2<47aLhmK*|Gi5^^HpInVf!=wq!GUFylbL^{Aan=`@w`9~T`U#`kz<}ukD`PG_uBj|1bL^?CZEV4%LrTeI4kXj$m&OK}OAEKtx~hn?@npXoKbUm+yii`U)yzl(kOif-Z0!b)@J<t)nXgHz)leZb#NAYqloC(HNOQXys3AusBQs}R%yf&7z$!ObiC=P_P7Vs%GV|ws>_bsZB^1?CEqrK@Nw{uJ;6|#$MA4u{nwl3NFcn1i!q{9{iV_RYB%X?ytDCA~wR3k-=V4i?nAy~_lTz8h9%zv&#3?yqYOMrueueDqK6-Z|iAGA7YWc=A36<rg+5ps*Wn-Te*^^SO4|!6bESPrde{24vm~v02Q{(QdtM*$;!yQ0x7YPH;*Y!OzE_sU3M)S*Kaq_MlrHC=SLPpO!rxbS7`!0SJ%(#X8yX>S$StRf_Qn+^}lqVub6-q-HPnE$PoB1ZxOYB0gi11dGI;hgAYuTJV04vxagArv8w$>N{l_Z&V5F}*7s3bKQa`a3DsxD8JJOd=Bu|CE?icGKAuRl>s!}YtiD(yS6_JF+|2DXU_TiR0IGF(}iKIk;3B-5M$8l{bl$?8vNLo`=5Ta-tf!^^(t1++{gJ#0FeIcc;t5$b@zQeihJ&NAR7?=MQcDdRxQBn%M_XzuW=BgbV;Sr-GiIz&&Db9VHdoaUI8w6$o+L4-lr1Pf6AnKi&9A1<fJDuq70rV8gl$H?-m-iY6A2_91BS(cfZ)5>T;nTIXHTdSdaWL-;@57<&^qI9`O7^06~dVF^~Yw4^8t}9iSAFyh)K-q^p<0U4X5^7?Nh(E6~`Z%(S<*|@8kkxvSQ~6p=w3n6#ma$69`}S2tlW4J^&H3^Qgyy6DVvB@gM^$rKze3}~K_VT(E4t~vWZl$Qz6cJox)c^$zu-zsi;`${RLPVsJ4zN-iF%rGK{B7X4*!tCk}}oKVucF()eZO_k!`nerYReT;+&mVcyy}XG2BvOh14+_JPKhhZ4l7SV_{lflBtb?Hr$eWEZ!>`$}M{46xb733lONa9@%l&GgXtyW#t`3hgK?p$azk-is~pz9bZRrol2(85NXX0M_M(fO{=0x%VXe_#z46cgB~i9epQDUsdYs;DH27iuKDYohnE$&Ovo@*!vzg=BL_ix&~q?(2vJjd&Y6aEc}CA`milE?#(N<_&P2M3QVfB4Lo(^>;pI#7hKIdg85LMr4PePWr3xe^)xrZQx1pz2EJh@IMq47Gyi1%A>FZbABb+?B@1nBN#YcxV@>pLwy4Y8YwMPd}pvuKcc7;EQqh1y2`f9LH9#_QUOmiXpCEVvdL_L*jM00kADBQ4hcHmD_5+C;ua@<)Ad#_rWZ=XxStFL9cdR4fyK%G+hNtS54L#PZNy0}EOw__9y(dEk+NUf_<ZAXJhmnGsG%h70YK0V5#)~Kz^TC=Jd#Z9FP{pC<~POo`WbAFgyq^{ek|I#Vuc4`<OR2YochK$OqCfef_f2OD!1vUx_c+irfz92zSEMjRK@?g;=bv!cxI;9XxYWLNs7MZY9rX{S(^%g6rD^bD@IXo$7qo@A5csR#(HWrwy$#?lC(RwJYk=6xe!3w%|I|!a<rhvTAZAj{E1Q~cyIZ~2syXh~Hx-4z0hI5u`GAryzA;+^<?vA$uU|wqNlc!lVIB}<)G&7!B&CQ24yOwdy4=)05kIh5X38V`^C9&QJgMQhT$q|-wvF#&xe~G#Ez03DaL<pk}Ti6~r$S+@Fz25@16-8rcoy{Oq#;CL^T3-@cnl#WV`zv9IbREF+lHoL!!8CbhIhRw4OV=w3`(iO}5nDmgYwOYS+!C!};K#zFwNk<cs<{;vp~{Tfd50DqI$rHQp>9UbDUPI5E#a<W`D*6$ZuyxjlnQb3P<67JF|=O5x<G!^qU>!5iLvV*?9ko@>aYaktF%<YoX*wRaxF!ik`gl$W?TUZmNA~xQcw~zKrCJZEA&FdghH`fte_eANC@d~V$e%9XwFMDm#IU=B;_KAa#}1t;I<LpGsj{2O721<FX4}JSDAm>YIxwZhgW@B(wZ_R6C`g@3x<hWC!1JnAYhS7I(7u)-CDYg7j;M?^AS{mJE>r9;mAHU))Tl?JpHyr3)>@Ldn*dx%kyKKq5NnEXzUxRaCQu|=-`xGas9QJ%J3C(n*R0)cLeOb%L>W_h`6IcrsqgC&C&q3+5rk1BI=UK#-0)-rNs~7ZEF>Xt7l%Ou8^v#-&@i_qs8NhdsKz9R)>LWNdf<WMy4y3Mrl64R%MOXqWAfI%=w5;`Z5)4g*R9Ml8|Hp7n<Ivx1ijzu3~BTR7c1-IB>ph@<z3Wjn^jpE_L=a#eHTBEzQDZbdyRNB}_}R(mGicpQWCbf_7f(RL%Q@JmQIaTIx@7I4S`OElnaV<~)Hyu0MXvQ%huz@Zo9R2$I^UyT)bSTmJ3tQI#gPlRSIe0=9ZSE^xSe8ipn4ns@1MAJwpJsZ~cGFKHsBsoi~;5iOo$G7BH9hRv~WiqNcr82=frYZ$8VMRdFa*VpwPle3Vi_?Z4{MO{}4fLbNkQ)o4f;ge`v`gSC8DpfpTp$5gE*TPJR4*?FN+zSI;#wBf@9?5N&`3c1QKXSCOa}^eYXj<p4G3BB`NML+Q3A`w!yKvz3*RoM2PjruDYlwj@H{`8|R@%(n*jH{}d-;dy<-5Dv4?VpMe<ZRLBLwvEX)w6@*k(F7q3e`J08W9ylQxdb4!9;Z_!}}!Y&bxlrSZrqa_w8&q2GN&{%iJ)w=@nj*whIbZhznZ^Zx$;TVi2%')).decode('utf-8')) ,
    '5C-11S-100L': json.loads(zlib.decompress(base64.b85decode('c-rk<U2j`ia{MoP=7T9wUwPBm+>EiBks;e5HUnWWKsE>vY#t_g3--Uqkwo(Hp6cqVK8Kcepf{SP?mge9ySlpim;XKa_uqd1+h2b_`KMn_e)#(F_QS8AZa#ncdcQt-Se*RFZ~ygg|NZTMzJ2`LZ@>TNU;pRZ=U+~K{P_64+J_&${^i%3pFjR|b9=Hld4IP#S)4XsfBtd3{xtcI&+GMvZ(nbIT;F^-SzOM({%L)C_w&hOx%>I2`@0Wc-+$Tv#l^$J-xt%4efs$R&tE_7-?W_c?U$3y`tz5^w*Gu~|K-QWPrI*X9}WlN^ZNF7|JKv_Tepv!yb3gA`r7@c`Bb0=%w8AH9_-<+C69BmJm~B5ugJST-QK)kN8^e5v;70`wpqK$Tc7`AI-X5C9>4o}zZees`aDy?&(aZI-^|~?UmiDK*7x&8H2>~!^}waOoG+r!cVFj=s9l_Y`v0AA@Xf4uY%1HqIUL~GDDC_A{^ogUZh!Q&GbdfQ=5jw=?Mt_#F#T0JU10yA$pJf|S;6Ej&tnh9Y%(0pjJ3bfXY6_0>ChcKcfRxXL)cD}x-P-sg3S;ft^8~`=z=z~=+Mb0Z`)FREah+Vc?3iFe8PY^%H~ZU#N9h~AHJQvpV0?z;P&I*^Wf)S(n%ltd^+JnI<WoQ$(x2gH~sJekDc8rXMr`D9HzzvGUloCv(?$Y-@OHMdxZS7F(byb;Qih0?fU(fU;nhe|MKzn<G&uB34;c&{1Rh{l;3frIoRIXllFvrXy=H`ejHro=MRPj_^Q{xF~9RZuIk=xYX5cGB*46D%*Tll4i>Ip&neC-+!MH0&r91flX)Mez0LZV4j^#s4TF@qD)3YGKsFZWQ~E&W5s3Dqx=w&U8aKJ<K*fV9*}lpKqP}@P|HRYjbA1)yDSaIDmJR0t829@}w#Hz-`CH(G*p_+wtjDFMD#6XZvtj-AwDC`q?|ooHtyGY^ZWzS2LVGwL!syExEdK4(yT3z7jdUEct5!NBEB3?gt&;=GzdOaY_jJx0LWHb$-3j!5?J_p#MOzsbZpVaBq~kPY?KjLOYIzWop<s{EMZZV=Qf!o<SIJ=HkfC$%&Y`S74siAM*tf_213p#<u=-4$AmBd#$zcdTg`Cc903`^?x9@H|Smw@Y_)5`hH1L$Z0AywnRX`9Am2%ooqUyblEW6-?vH9`l{$HYwjl1y;v_OnvqoLaMr8q>>vFM=~w1d;eAa`VfE+B;u`mSS7Z}kQp8CAPMnT}KrhX8!#VA<{0=yp){5l?y0_fJGu&GdZ}16{{p<{S-rpMf`2BFycBTbk6H+3@z*UXbWKvz(rPclEg4TVv`R6Can3jrDBAeExEOv-x3tfBzR?$&?VLv_s+BAr13%INTH(bX(%*!x5mT?}#{A7o}k{&T*&e#Tmm0Rg~#i_SDoAnF325oWv27uJy6|u)DwW$7!J2_WGUdP|S247}xA_Mdl+=Tm@0Tt*@V&SvCLAr>BNi;;HSGM%ZlvG23}u1rt0P_wFr<^-?jh3kN7V?=)i09ty{&5FATv1Y(t-<5lsTsbiiYFO_SBK@*cpaC3Ke`<Md*4ZY|8e4e0h=i`TyvSGJ(-f?e@ucf1NAx$;oLkuFQ%!fL<71EC4S+UphZX!ek4hD|T1*#87OB!t`l%|L|9a=C<y{{$Asf!Dy`!0P{s5aD@!eSCs+k7g6o9ieU!fPVHNawLvuZaj!oDifAvk1C(fzkPPy3x@$KP@WTfDJSHqz?)>by~D|o+neIcLASdX5Jplin=gUW0xKiDK@{BI6$|`tumOpYMW{zcS>`D<v5p4us}JxAjdR#A-IPtu{Onx&iWpF-Cp3M2;^+aU;=KQZCzu(g6Gn-x)NWD;a$6^kOAK5E0l4)ckv>Ugk0odk*lAr`Vy~<<(Aq@3I0J}=$I}i?Vi<5`Zn+i2G_U`8R{1XVeu@dwv%+1gUh!GclKqLhi1b<`M)ZfOstm5Shfq2j1qZe^yBXIlH5wGxftPKp<d09USoWjMVo@21cJ`&NRrIJk}fJtOqMqxr#@RyBQPWNWN7P6fbO7K;kH%XY9qYEMs)*qZtvZnKHmQM&|!e?1iE73&sb!4_pT#57H2t7vb)KU0i+vEF5N6}F(5w%7*LUY-CUIlS&^eHb>~|Lo-2No{`B;eWm^EOkKldTVrh-P1E-ac{FSXF&<}1JbQH+uAR}G&?-cnY*>$7bVRi9nepd<J)%XgnZ@aghnZ1I!aDBL2YqbM=mjrRRpjGT`m>F6(1jPrA2lIe1gArLpHQT)%NYr}bcfu%856@A-Ru@SmqDi6m<`j@5!PEq%DH&R{4ZvnHXScn#5F*c|)IbxbE3akDR-g^gRY^6eRi10UY5%;!kl#zM#qSPDx7m5fT88Z0WrfosIajO;`zJ|hAVMQ|v+gxSa%2jbePE7ze5YRGPwh=FPWydp%=c?dcSP)vjeJgpV#|5?OqO-85`gvW$nP{orh#e#t$n6@(}bbw1@LkL^AL}k#J?=cOXCZjNw5I6HPj@*qQcSOd_;g%?^X$zeCRn>u$lC4<TnYh!V{5dg;yZ5uqz-4|HRo92ZXPck(%1{FmcBckp~*MDzw}p!8`aE<@>i(gia@&JK}tFiGRrhyz0@kiQ!7H0ODlRAn+XV!TN|oBy7b(fh^cbd)5crB$S0oKt>EhYaW-*IwHO*Uv-H15ZiiaEf?Ab@>jQSAkLN)>|$ZBI*!g9KH`s!wWCS1YNIx!Rmf~U&d~5?)eoLyyGD{s$BnHq_p&SQ1zQCy*4NOiw}mxecj{QdG5!4RAp0@g`AT$L3_|PCapK4y!0o;(uX*sYm3&o7&<KJD7L~T*>sB&j^su<@o=060GzY!(bruMhoIe~^EN9qLIl=5*5*`)%;<Z@^anEe>QjsH?^4K}kbhxH35yS4CY;f)(lJo@}*#@8l$|VEv?{aC}0r-_|yW#JUup}=<1)6(l+z1-ALE?vkEtrq&<%3QlWur;`W|H6*B@E-%UZF^b-U0M>yo(uemcUk}9$dui$mN4U13*g&G*!d9Bq%oTnGV5F5w>Kb?3i)W@`kaB8D^+W|F+sckdZnm^hVkUddL5OT>FR|dDzL*Rn*D2^vUwv$fZtOr{*xxgjn2#cMp2{4JQ*(NEPz6z=ZWIxpxoa^eWi|R7tlz@>OV<qq>r*O{q)!@i0+H1DYmFgp(YNq?$#KmW3e7>QzmX5+e$@MMfEN2@Ys;B!v9aMw(*;JOA{oijHu|<h39rQMp{pDLy&{VvpX_TxzgmJ%>NPo{C&+;N%JTiR9-8e+x&dBA(Ow(^U|q$mIyWZf&J$Pxac!%Cb{aRT$Ye?S_opU0Z1_6kcH21=Lh9bmP3xn7rJm&q?UR64S%eooUt!bE#-UsFPiz>oWDJmf*uuCMe0hwf@H}5r8`)(Gf~IIr0LLrh&1@sMEm~4kvVUE*#Kew$ft`zosn-2YY>-Mb+-%eLtGw?6X(Nbz{P|qrFvKR)m|CB!(!>OH27gu6Yn~T1b6T=rH+u=R~;P1y$Wa80yiIjKaD{9gEC3rPBStTFf)eVyxI6bHerGc&MPqmH-#SFFJd*vivE#gU-5v?UK{gl(MD%%=q<fSm6Xb@DiT#eb<o(2z7q2)`$%P?ExQ232p!oG!pA^QK93*J^=rL(zw8u-!^+X16SQzqr0P7Pj;!(P8h<QHy)<9v<GyX8Ir;xNif%{^q}90?V{LE#!#Fl3_a9*?5vggxHAxxV-6K|l@cDI23{+>mtdf3?DqVjLY1$jIU%AydXOVDz~mWNqL!*EU}$iWoDG1raG*ifp={&iHN8BZ!lPtt5w41m<f$_`L`VvabUS>MAW#XZNQpstF3&i+AQ+?x+@W5U&Nwm07jp&iw|s}pk&dC5921~O9QLT@92PSY_s6+`l88beTzCd_T)k1$*fm=wd_(K<g__hSuSS+EBZek&?=@TXF?Y>hFL6mAv^+I}2Ed<Xhg_jfibbb{!%FTFJBEWsmlr#Q?@e%~IN*!k%D);-O8>RLNCx8y%jJ~n_n5Q|$bXkFCb2OBaQJRS6J~#N=mBI5J2D48$E5Wc`c)`epRK`9h&R_O-VPIwGDQwbbue>AJSeKa7Lqa(74V`H(_%CdNwS6Qg=)%9yGm5K3u>MeN;J^?JqC&BBpc4X=_CiGDTeB}wwGkF#Dv9yZr3XtV0MA~_Q&dBkb|v?wrr7&8PPf`T#5|^<8o27G?Qe{IKjipPp3(;9K@g|r%}0!H42qfU}z7rjZ^8a3Zy6kE9z!14V>mLOZbz@q0-09E)cAx-HjhIF%YhJ;ZoK@tzMDXB-%EyEOU`)#7ej!PaL&Nf)tDab%k=(?*f>qMUyG)UE?75gcK21AZ;zEWq$bh&mBYpqb)Ue|GM$&BLHCaN23fBys`k8&yx2$9VI8G2(v@S*3E6rXj3AiyFZ11qwJRG;ud-^yfEj*Or1C!+3yg$Wn{D3BrrnK=N5<MJ)ZxWWqb^EZgkCDTWF(JR}U2mu`+mSJC0hfhf`B4{;ViTXv;KFi|A5?Q1xCbQw-j_XI^Ck+@~uedMSc3N=<4N@cKn3z2YRve5(XS?y45g3LJ7EWmN5Fj&-s;oE%%!d=(h2XD)u_#WZJxW)$Gvr$T6o`=b@BQ6kX;Sd&UdN#cpGkK^A&-po1ndtSSLgY2kR5uilSqZq^MNPuU-Ni`=K>>p~dxnyWH0;zmX^ae*L-z$`89aGjd<^%#e&j~L|q)BTj?Xd1IiDRd8RPsD{CQA;OcWqc`azIksN>Y|n#EROpS--34QJ~Y{Sp>Ro=B$E`T5(4@E9utLw7_T=2*bP00kI@Lm>_0rTIj1x&?DSGwoL}1-iGRc*dmI{Gl#1yXRfGqC}>>;6+t<566$HntUK~7qo~ksvgWOHEw92mk!#HRaPBPnX&aGSDLULdcSK$`fNZ!dm59zZ15eJDXvuxzou|(I0Nvk!KfU&Hx(mk>=*`M44xAtu5_gscEeB20g}?^p;Uc`jDD!EE6ff${m9Xg8#mmqTEKVB!obm7brtC_<RYZV?A8*=1WEpfupCDIV(dW|`6yjzQK*^E)Q<9d3W%D=1mk=kYuajfYu7hWXFP#cGUNyefas_VMt1fU1&CRQ!?;8d+*IP_Mn@Y+zNoNBNyB933YX(%A1In&VRQ`1vo0LNDqz-fk=f{qA4_;as&aUN4p(jmD--%)|DMj5m$O?JDt=}o+Kw(Jgx({S(qnb2KH;RY5pSom$&l<vy4CKSm8%v6IG7>@dN~Cpu5dw0D&tOW=7q0izA}1Q_qmpmWHa9Q_VJvjEJ}_4yUpK8C4`}@0bje}h*3x|%P_wdO<4b^PVd^H~$>>pmbWk{r+z2A{c4QeP7_bN<SG-sk3MVBYJbD%j^aD$lzn#Cx#DwKliIl5Y1w@!JI9`hGUUX1vaBasLn>Qs%%;1)M+b=vVWYUAHd2wuBya!T<Q9A*Ms1`lAz%>uj0OABjn3WuV<9iU-21e`HUG)-d561HpOnI1oepV&wdCBJ+nd<={Hs%m_`nD+OSw<BO1{+{zug4Dt=9nS|HK9j@1-9m<Fg)aT&v0rOsKrtZd)G<U?n)eGMFHU1ld&(fqJf>>IZ)XG9+#^4*hl==sG~&Lt`*@^DygOZfLCIUHlA^T8T%u#&l9m_pR#1pGTLPx*0;Nu_v+AA)<fBv1_7|N+$wSilqQ@RRi985*k^KOH8>h9no_Z28qALE^W5&faCFz`_7N`h!}XWxv=s^!UK*7gxF{GqZW*nH)h_h7i<7@ARC0(5ss`Wj<IT*(yYAB+b?H}W9olNMfrCtHO#rMUgIPrGlN#aI02Ym;K&X?6Q>UeoMmu=Xi^HhwI6pB+Ot_N+_iAS;KuypjBBQZ8ogT4}Ggqw?{%CT%d&4nl#Nhy4ivvWZWqjVv+LJ_<mq0MztB3Jw44kF00{6`2o5muAw33|6yol2Aa4Zi>F{&rlGK}p+rXnN4QLA8A)*#i2cI^lhnT-li=*3laUA|nZb$d8eGNGu$ro%AY$;-J!po>?v94D$t5Yx2!O1O^D`0pr761!}iz`r)t%~z$gCMrOIZu@N6Rd(SBK}M4A#5<HEu*a;8W2ZQQH|IhsJj1r!aw*M;7C-D+O1zYarZdm7oD$DCmdwb+AAKk1>Y=3g(YEk#sR(5wa0p#w{fT%b0h6fpr^+r8*W1m1tx4yrOA0P8?p(37jB-A<Cf!N>d`2yi9O^)e$fj(lbupwmQmD&YDZzb<+i#R2$az;*qMOTBprm2xO`bM9vYC7qM+V+G7ruBpu%*X9nZh?EEbCk(@G!|!CUwM&OfA={a0LW_&AAh4l#0fR!9~u=8jZ2?;wnT4e1hj`ti}<t=TF^g5k&<|cJa0+6z%jh0Nf6epbi5Ju!(P|TX$s>jcM>5go84Nx}a2(7d<&YPFWMojTq3wO-jQpU8)7GAxlk!o47BTe^sreBd<g!3C=8$(jsQ6Dj$h*<6}E;g(RKWJQ35ZFp(G{QDd#@DGn^ITB4j}a+KOBA+ETvC^lhF*#ula6vS#v0#R0Z25qS?4f>KIgv*F7Q(`<P2+#|TPDvBJY|<W*$y3yCZFOv?SP3i4hbQW^(g%EXbCgdDg{#ifnJ^o!Ah61jRd5WkN@C>lDWbv~9)9hqR4K0>3rZ$bULq1LVxBsNc_AzvhW)8I5Bzq^3vsgrb7h>o)Iz7X0!)#Zc%m2KGL&?3lI`BSDJuNo&n1KQG_Jmc4l(460S{B}IGW^Yony5qpeAxaZz#F2g38P~t0WMR6NOg-QP_*ZED|H>!nk`y4f7U%7b%qL#d}%j(N%?mLLlrNEf6%NLF#al4!gg?vO&k=q)AM)#83Xdq%_zKj6aF0GRLbw1XxXqOU(rON`gM@99#7vWKOyq$Yj$?{Rn>R<uTTtq4WfxhvCz8)|XnOEmA+8Br9@YncGDVeqz>QI{X*mP$@YkrL#i;<A`~K;WtKa6Ujsoc&NZ`5v3ubZ#icLXY~jN!+B?oFJuL)g11QpssYtMY1wdFSxr$#neT^M;V6>%YZvN{mRVYePRcRsp}gnWrGsi=rQAuQJ{RNL1&g<mL#YlCucW3W9F{{2fShWq2Tn=nyw6IYr(Gd_zH<We=EB&Py~$|wgmISz`!#wRa2$&2W-5$ApwYZYI3L(57m&0`Y8>dWA;;xu>~VoswzIS}ZM!`8Y3>=isp5&HCsb?a9>f?T(}CIECSx`*@0;$`z&d9<sPh8F%@Q3HJ-zJ3>YGJBkhx*U$@R;kmcxa%MQ4$uUw5Vb=^`PDBhe{sBta8q9-yz;q%32D!3a=(CI?wEno05?G35-En_N%W<NijI>tz(?Vz=wA$(5K#3s;lqv>bJ0=vlle1Gkj2n|exJTn|r5v`R~2D^7LvGL_M5Hm{YGM$QwMlBt+9O?o(+Gud#s3|c_TNF%FMSE)wJltvT8^dNbDmo*tKsyiv^v;$@SBk#{FO5^azgF?1|2=lbxC~(929R{^S<%E3B&z%d(&KZI#(vy9E9_{;Tk*vv6?Rz+>+QRXyB;XoKg&UP}<vD%lD6q0QWegaxD7rC`XSHd|FG-rl$<x&n?w)8FUiMwHRC$Y%dlJH70uR7y7sV5KfP_|V%_<YF(7rp1dKMM@Hj+`7PhcFvIqKZ$r5-d?Ro#O=H%*$eDvH@^{D;J})9YZgBlhDTJVi4+5zi`~<Gh8d@JqOh^grjen~R+)^$aAH+n8e&>s!_8T1is$nbJE1T(x{JBV-9YjY6hq2fK?SI<){em|`7cpvZh7bGB#MaI!-t7pCXq7zw{cxffPrjlz#|GOa*UI&=K_G&#4G<sl;Qu#ZN~Lq_*li^n0YTUQTr5LSo*9f2l^gles7S5C{5lyLG<qNrWO3=$LKa&&s-Y)>Q>+?+g#>Qqz#klD9Z2biyb*qI{VCXFP-Ku#JRP{gB&{<l@{C|z1AzcrtlIFgl5S)$X+1>-QPl-Hj0XAY)4Is!ZyMM{$FPCPw3Q>-KRD49x$Vv2EmfifP<c5N1`PP<qrG6iEYi0wFwl|p6lZlc-_6gz5H^ERPwF!BkuVWNO2Z!4O_>7}I3%DRl)hibxzR6HU&Ou1b8(Gjx*>dIzR`GY~2;25;*iWbD&x-ougCgcph9vUEM-gy_kcU(0yN9g=0R6$u8+t7!F5@uJ&kucNiV|vgA?8EXHg*5xCTuQUk%aN?<mg(Y6R$T|`Xi3a_A+%eSIZF<Kjl>!*ky5TlWoOd7W|@c+WIy(muQ=<A*IVjp7Au9Za9|wuk*qOSB-g~3Z4oRczP7e?m=2jHS2?~+b7C5EWv)2fPV1_o*~W7Kb68>0<%>eqN#Jj_iWz3HRzjDP60CQBwv!5HR&A1jHB{dWQ96~}rNn`;CTg6t995Fce0VVdE<OUQlwl>T$>}>eR%olwU-Xd@MU9nEYfH88p+P3Wy*1q%sXP+}i56+bUI5J0Ktp|VWtmGXYLj>>W(qH5kl?uswb<*fw3WG;O*uP3m5n_H<AAASY6%5#eubp&KGt`lltwC=YH7$cH<hKU+PKuUWn-TeNtIFw5P4F!7EHVKzc;N?%*3bDsZsjXRr{J0<1V#S5t8Y4Osd`~0vpXQk44qHb_68G&<z=D?}9ShQSZBGS1=J5((<yCA|;o=+eo40nNXgH995VOWjs{|cWgqOP)4x}y&`&CRYsx8vKH$7Rh$;G<iG|Qj3{%k^++~9Tcz19%TiTC^^U=iqh}&e(RnKH86Y{0^)UugWHQEnS&CYAuGY20gb+lH$l3$;b{Ov_CTwX-dCPERB><t*oN`ZdVrZ02GA7wSVH435+H6rCaSpHgq8HFIkz}&zWaeDb)<p2Ussk5*+xrJUU6#CPz+>Lupm?{&5t>PSA{=Qku+bI-)rMi!LG2;(qnzTS@9;DwwIszwLlGi8!sb`FL(Z%LCh2oIqgJU4;x$z`Cpt!!Cjm!va7!PN64kOq&YXot>(4xH5#Cx2^&{(ADwM#MN)x@yJ;e}}{MO_9yID(TC39V6yZn4jqX@=6<Own{Kb0^UYrOq=QPSs;T`Z4<xPk20gA~lS;-kHEJ+O>by59FMI+{e01u4(hS5Gt_Z4&!F0H)7hzRD<eY0J*-)KKgztDr-EMYq1Je7EXT&SL8qTxn?y6Rqqj3y(M<k3j@J&A1?mQe3uwNF7R<f@iTsh25zHf?Z&#$iiDY)0B-van3F({5n-08Ez@Dpz4@}AB9MlHVDY<vGA?0;KYlep7zD)y&9w3qGwKlJ%OEMp+t&_9fv(rH7Rsf-cfXDrFw~+|744<j-uM}brhGiWMU4HH0^K%R&(04qN}u;2F`#Cgbp$2p(5<pbv_8St|(VUqHfhCg1z(bssfh@38`urp@DAX;7Ew77mMm5NKNTFryA1r89lFA>sM9T?}dap6R9i8Jp}d*$-J+}ov+L*9`<@=RBUB6gCzr%DwdQ~4i7}#h8|n7c#-THZHVOZE~BDc?*4krO(hbIt3<!Zm3{57V_#9%9z8vQTNf+h75*d+ht()qV+HrPvL2_P3rLx8(f2U>RB{qc@+sGRfT3bz{T(>qlnuxoiX5mWhaf@3^aaMNxMli$RlKvnwNm;?mT0>}sGJ|V&_va^V-z3JWzZN(t*cvYMx#qtC8iup>SzH#J^Z8Ax;5mnBg4z6%2C{Gx{zRwZKskTS2bUWsZr{(p!zSJVlt<OO+t0Yc<ITgdTSzWUU7deYBWHwe^7vgRxI^}4T{mLrE&IyMVFKb&GhY*IxlY2L!@S8x>1=_v8o1KEYPk*6+6`Sq#%!;LhRz_90poj;IpO<=37eVLlw?w*-;kUpljFTgC3xDXQn>A&}~Q#ZiF3p%{o%?ZL{vLz`87LtA?YNN<J%WN+DOYS51$%13+YIX_f7)8XUUQPFfmI(dOnuo2AP*>xUPC$+@0|wi5&xU`=8<6NVJCEt69;FT}Qw;0Y$?mRStA6&!qZx`0iG|DnqU7oe>uNJGnY2AMTR!B)@TyefgHNyDwG!VF?m4n~I4R7TSzpyjQW0@n4)#J;+WPYKL36g|G4Tk16o2w8ZvR+892H8-M!RGDf!@6e({_^aI~)St-7#*x;lrSw%SgU!6#4L@{+k|Wnf;;PPKGltd+WEZpvwdi{rLSnSLM}xDsfja)ch%7C~Fz0!7wp`08r)<T{h8b7Lf~A-zH5rtQ4G^OjIrnlwV?wRiEmqJBd?bXtI5Fs@mb<+u)o!Mq6VsrJu*hlA`GE9BoXi|>F-!Mi!&DP4O9-djSLUBKrlz2tJ*`P;G6nM%o?zgo)o!t7K|n8+1nvl$>kay49ec=p1XU4FD!f}bvQLfm1TGa%@GTL~_L$k;io*Bu{Mcr2KH32i`-V%R9Rn811ly)<*@A|rQib#<ya}-Ot}1{FVCas<j$R@MHOmIvXa^{4h$v(x8+%GPlonBhx2;uZuAYFIx<aZ9e{T&4jW3Tw>rpk&THyvRJ_Y;(8iKA=ETxG88<kC73)AQKG3O&XdCgSe72aS4NJ7#PTxbHN-hy(=!-^&1Q{f@w;J}Hz$s5(8H(s0YyVTj!B>I^#v@{E#(M>9emoV|oO6z19eU_72iq3hhQ$_L<mWZbYYRN*$@u&nSw8V<E;PM0tx%vD#PdSl2!pEn1BS>nj?i!bQZ~3>oM^&2GPV($=3)t%UxWM7=X&Bb5Yu=^5eN@9XrB)q%yrhYgrgryX#;ACS$t--Z8aBtiy;4J$D_SiPxxG^#`7CgUFDu?-auzaqAJc!W$ni=6P^*-C3ZbUKdJ=6*-;PvKr4lJD)SwvjT9_&EA;4jjdtso<xCYMlk=%BfpFqt2BS#xMS7G&trgiQbQ!W~W1jeV7KzCBQ3kP0*EgNO>ME6Lxh8Wm#L*9z`qs`onedXq@mw%XEez?E;)YHrGM+9kyI*oG>|LJKkxcb~?Iyj;0B$`Du`KxUlnH_LVtnoKwn%HoFK1<_~Q{>vWwnM-BhWyv;8+B<!wrZPJ**0{z{r&LwhyMdRBPeY')).decode('utf-8')) ,
}
_CGR_DECISION_STEP = 168
_CGR_FEATURE_MEAN = [0.21153846153846154, 0.25, 0.23076923076923078, 0.23076923076923078, 0.25961538461538464, 0.22115384615384615, 0.34615384615384615, 0.25, 31.807692307692307, 36.93269230769231, 61.92307692307692, 159.30769230769232, 268.0, 50.60576923076923, 196.05769230769232, 186.93269230769232, 91.39423076923077, 9953.39423076923, 9981.403846153846, 9985.673076923076, 9976.096153846154, 9993.0, 9986.192307692309, 9980.76923076923, 10011.384615384615, 10044.25, 0.0, 3.4038461538461537, 2.798076923076923, 4.855769230769231, 0.0, 0.0, 5.471153846153846, 9.89423076923077, 47.83653846153846, 0.0, 1599.7403846153845, 0.0, 4.0, 2.0, 3.0, 0.0, 0.0, 4.0, 14.0, 50.0, 0.0, 975.8461538461538]
_CGR_FEATURE_SCALE = [0.4530468842072974, 0.4330127018922193, 0.48498154665071097, 0.4213250442347432, 0.4598325846622441, 0.415023881825139, 0.5145803138561411, 0.4546765545206496, 1.0748606624986254, 1.1624280215675473, 1.2301681224399605, 11.787441918510078, 1.0, 0.8368699608427508, 11.828767836115055, 18.790988236527795, 1.0599189848464232, 12.148455322220045, 16.83261442212775, 10.523273024488574, 12.278435780143205, 1.0, 10.657316846508024, 11.881814457097837, 15.791494602321478, 4.8570923559107, 1.0, 0.8147584678582633, 1.0686062786163213, 2.132253544378528, 1.0, 1.0, 2.3202981024048746, 4.422230270138612, 7.028938229429518, 1.0, 1165.5829838390905, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.63230568135445]
_CGR_CENTROIDS = {'10C-4S-75L': [0.07143626545971402, 0.07040856941336894, 0.05222534832778028, -0.48983318150868593, 0.12495370713784149, 0.054812732150504695, 0.03828326084826999, 0.04023229864164576, 0.2129513769991974, 0.0474115982529001, 0.07244391823252523, 0.11977322112719123, 0.0, 0.07762553274396475, 0.12605608637951574, -0.38581076136970877, 0.05376707444541689, -0.18101962428002788, -0.0631145413641154, -0.07554951208985256, -0.1250304406755845, 0.0, -0.07525944864183244, -0.12221682165986922, 0.39961426866311794, -0.05649269641095613, 0.0, -0.03166260232454134, -0.050696393177363894, -0.023867322027866993, 0.0, 0.0, -0.04538218063796278, 0.021159935847274908, -0.08257874541173238, 0.0, 0.08952170946644403, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3697819229395356], '6C-12S-100L': [-0.19101436199010421, -0.14433756729740646, -0.21808918607248975, 1.8257418583505538, -0.42866771775247214, -0.382276425771304, -0.0654005706157943, -0.27492070738457874, -0.809586152003604, -0.37223148415573815, -0.3947240334222033, -0.5934444772709859, 0.0, -0.2757527950182878, -0.3641708390404173, 1.736460439524868, -0.4898777893916179, 0.723817884458631, 0.5997971315069218, 0.41117654810006815, 0.6131763270710305, 0.0, 0.28691014368155454, 0.3665912514033162, -1.8964079169689723, 0.4632401105698407, 0.0, -0.11211439641280838, 0.5398836676030977, 0.44869465535798664, 0.0, 0.0, 0.47034738886138283, -0.4566091419675165, 0.3077935056255475, 0.0, -0.24541614675362683, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0038678910167254], '5C-11S-100L': [-0.4669239959758103, -0.5773502691896258, -0.13217526428635742, 1.8257418583505534, -0.5645867502105731, 0.27029646266657853, -0.3488030432842362, 0.1832804715897193, -0.7514390803127373, 0.34465878162568525, 0.06253053994807488, -0.05438208268233646, 0.0, -0.3255414941188113, -0.7516442764122887, 0.6421858999863587, 0.5715240876235453, 0.5437538399377636, -0.7368936187084483, -0.06396079637104495, 0.07361248370964664, 0.0, 0.26345208161953415, 0.6927198922764441, -0.4043072264785281, -0.46324011056984055, 0.0, 0.7316939555362242, -0.7468390735176179, -0.8703323465737833, 0.0, 0.0, -0.6340365682448593, 0.9284385886672829, 0.30779350562554747, 0.0, -0.5690203046983958, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.28996142920427626]}
_CGR_FIXED_ROUTE = None
_CGR_STATE = {0: {"last_step": -1, "route": None}, 1: {"last_step": -1, "route": None}}
__version__ = 'current-gold-imitation-v1-rank06'


def _cgr_tile_counts(farm):
    crops = {name: 0 for name in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")}
    animals = {name: 0 for name in ("GOOSE", "COW", "SHEEP")}
    unlocked = 0
    for row in list(_get(farm, "tiles", []) or []):
        for tile in list(row or []):
            if tile == "LOCKED":
                continue
            unlocked += 1
            if not isinstance(tile, dict):
                continue
            crop = str(tile.get("crop") or "")
            animal = str(tile.get("animal") or "")
            if crop in crops:
                crops[crop] += 1
            if animal in animals:
                animals[animal] += 1
    return crops, animals, unlocked


def _cgr_features(obs):
    seat = _seat(obs)
    shop_names = ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET")
    products = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
    crop_names = products[:5]
    animal_names = ("GOOSE", "COW", "SHEEP")
    town = _get(obs, "town", {}) or {}
    shops = {}
    for value in list(_get(town, "unlocked_shops", []) or []):
        name = str(value)
        shops[name] = shops.get(name, 0) + 1
    market = _get(obs, "market", {}) or {}
    prices = _get(market, "prices", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    farms = list(_get(obs, "farms", []) or [])
    own = farms[seat] if len(farms) > seat else {}
    opponent = farms[1 - seat] if len(farms) >= 2 else {}
    own_crops, own_animals, own_unlocked = _cgr_tile_counts(own)
    opp_crops, opp_animals, opp_unlocked = _cgr_tile_counts(opponent)
    values = [float(shops.get(name, 0)) for name in shop_names]
    values += [float(_get(prices, name, 0) or 0) for name in products]
    values += [float(_get(inventory, name, 0) or 0) for name in products]
    values += [float(opp_animals[name]) for name in animal_names]
    values += [float(opp_crops[name]) for name in crop_names]
    values += [float(opp_unlocked), float(len(_get(opponent, "hands", []) or [])), float(_get(opponent, "money", 0) or 0)]
    values += [float(own_animals[name]) for name in animal_names]
    values += [float(own_crops[name]) for name in crop_names]
    values += [float(own_unlocked), float(len(_get(own, "hands", []) or [])), float(_get(own, "money", 0) or 0)]
    return values


def _cgr_choose_route(obs):
    if _CGR_FIXED_ROUTE is not None:
        return _CGR_FIXED_ROUTE
    features = _cgr_features(obs)
    normalized = [
        (features[index] - _CGR_FEATURE_MEAN[index]) / _CGR_FEATURE_SCALE[index]
        for index in range(len(features))
    ]
    best_route = next(iter(_CGR_CENTROIDS))
    best_distance = None
    for route, centroid in _CGR_CENTROIDS.items():
        distance = sum((normalized[index] - centroid[index]) ** 2 for index in range(len(normalized)))
        if best_distance is None or distance < best_distance:
            best_route = route
            best_distance = distance
    return best_route


def _cgr_route(obs, step):
    seat = _seat(obs)
    state = _CGR_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "route": None}
        _CGR_STATE[seat] = state
    state["last_step"] = step
    if state.get("route") is None and (len(_CGR_STREAMS) == 1 or step >= _CGR_DECISION_STEP):
        state["route"] = _cgr_choose_route(obs)
    return state.get("route") or next(iter(_CGR_STREAMS))


def agent(obs, configuration=None):
    global _ACTIONS
    try:
        step = min(max(0, int(_get(obs, "step", 0) or 0)), 718)
        route = _cgr_route(obs, step)
        actions = _CGR_STREAMS[route]
        _ACTIONS = actions
        action = _weed_repair_action(obs, _copy_action(actions[step]), step)
        action = _rank_sell_slots(obs, action, configuration)
        action = _terminal_liquidation(obs, action, step)
        return _align_hands(action, obs)
    except Exception:
        farm = _farm(obs, _seat(obs))
        return {"farmer": ["PASS"], "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])], "market": []}


def kaggriculture_current_gold_rule_imitation(obs, configuration=None):
    return agent(obs, configuration)
# --- Fixed current-GOLD route probe (generated) ---
_FCGR_ROUTE = '10C-4S-75L'
__version__ = 'fixed-current-gold-route17_rank06_10c_4s_75l_market_aware'


def agent(obs, configuration=None):
    global _ACTIONS
    try:
        step = min(max(0, int(_get(obs, "step", 0) or 0)), 718)
        actions = _CGR_STREAMS[_FCGR_ROUTE]
        _ACTIONS = actions
        action = _weed_repair_action(obs, _copy_action(actions[step]), step)
        action = _repay_shift(obs, action, step)
        action = _rank_sell_slots(obs, action, configuration)
        action = _preempt_shift(obs, action, step)
        action = _terminal_liquidation(obs, action, step)
        return _align_hands(action, obs)
    except Exception:
        farm = _farm(obs, _seat(obs))
        return {"farmer": ["PASS"], "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])], "market": []}
# --- Prefix-safe public-state terminal-margin router (generated) ---
_S120_STREAMS = json.loads(zlib.decompress(base64.b85decode('c-rlKU2j}ja^!!}Gap!=B==3zD!T=%TMen(1+PH}1Me&r7<&x#jGv78?<<RB@zy<&kr8>$CD{$WvDsAB`MftXGBV=7-~9TY|Mj;&t**ZO`p3Wg`Evhdb@^{!wtxL{wfpxs|Lb4>`_2FT<FEhmr(eJR_~sA4y#Mg-%TI4V|NP6R!<*ZSH~;UCfB8Ru{rZ2t{`en%{PoX&`rFr^|MkuH@4x+D?T0`7@-JWB{`mffw;$eIy!r0q@y*4$`Ss`T4~L&7|Ks!F@b2rckKZ5O{`}_RYWC|N4j(@L_~v4D`tzSYeSG)JcRxS=i~a5G-!G;U`|166|MbgGkKeSK%<W&_91ov={x;ShKYsf8`)@y;el`2yd?7v`K74rm*7f{bA3kpKRbU{~uYLS!z7=Qyv#$$x56<v3l5cymI-BeLUy<+n>BHOa4$*p|`8@ms@NKhklW)EMC)4$8#_{cUKRzCYv$@{yRPbl%3LoCizyI-gy#4v`X?}?2zdJuY@aV4Qhv@UiU*?CXJ)HmWKR4s-H?!HXt!x+Pe1T`H^f<ro-rgV0hd;WznUkqod-=Fr?Uz2R!t}4w?E?1?Z4S5z%?T#oa=-Rq%_hsy>{$CZ`i|YNJ6*c7&z;}A$0cmHNj;ZfalvH>AFce^ayA8RWzj<?-@J#B>T4<gCf`S}g!fMvut(Xx>5F*!j?)icpS_>a58lE(u6xf1-~T1u^s%2$H+;wh9{%m*n}&XF`olAP?Ch<w38Km5VQRcU#(e7h+3H;1Cm+E)9wC3)gb`y#@ZHA`9}eIB{N*1GpMHM-;r;(Ue<loCyz-YAN2L563+=(fTYEB|@D804k=Y*yukzik5dpsF^Kb0$e2&|CcALh3oiPar@0##&5`=?;E5vh3vI_46p4I))w!&mShv{syIi?E;5_=;cWvL4MDf>XS7U-w+16f8O#*gYb0r_Y=<YEF9FRB##Di?_6=K21UOs6mPRe-1T<6yRIBo83CfBeWc7|bt!3!ISHG9RDyacQVZaI+^4tY1HE{HMw9eGo%!Rgk+L7{swc`*1#o(N{BA{OeQi(?^8VNRLDIs+As+mH6TGt&<0q|LzpW-ZMFC2@$g1^-iGg*REiLzGy4U!sD0_igc2uZ2U&pL?aIpG8E!5dgvcfzZ4rK=vA^9Ib`TOc;`^o?+3Vg{n*!!{SAJs4q)|#IzhpG{wL=p{1kH9*#JsVl5fAe$zWMJr{OEbtkJ+z`U;R)L{tGmJXFdVKdGwsCbH}WAFR#yZ$JHWG_i4Sd;={ItJr9%PID<P(R3|(C<g7~v^B^RnV<_u;fsEn*xk2!i;jY-)1pjQDu+V=zH)Z#9?$5*qC8eS<wbw|MD(hee%~ZO*D;tmM}xl4z&BJX%<TuaG^y9H;q_yCL84t2Io<!=^|$NZ22(psd|Wv()_oB3`R7k>kAFCP`t&a#k|`lf>4d^>hcwLfaJVT9=pKllw+lc|Plz~KFG?e3oa0W_7iSD3R8gj-;;CsUG6j}AIEf=DUFYNU!|DB<f1C!YJ-mJ=Hxx5n2i7&aQjz%z6i-1kZ|m20!>opX=%>4ZR_dwkD~+(*1Y)-HaTQE(G49=K73;ZbVtWTD+I0poo7=+UQz(ulF#?H7(BoCfoM~d7AupBJ3WFvlFTvx-j~~AEz`#K7{(s(Y(AVeVyEo;)?lF1Cvo(G#JvuL>t!8|PK?IfgQb%uvbfS1y?6Z8@2oZsUh2xz-^#N^3qXUJ~7O|v5D~74>YYB7e;)3b9OCJ@gEp?`_m;}`}Kb67FL(~l64H00Z^V+L7L<A{L2vX-=1l_m5n0&k4=+QTST2!t9TW0i=z9`_S)1t-mKA9T53*;oT@OG&vYHzm2t~@3ZEW&N`$P2co+TobUkJ1udxt28}QFwp|YSjg;rNJ1$>zBRGbhJXrr?UkLe%-#wgFobug4!;~HXF9*=DpZngdn}QY5QHw>rx;E29TY~NCw&m_Wy0tKFwI-r%DrPUrw4yd!h$+gJ?IY-aO74R_zlA|3PN}98$?IlHy~F3g&NFl0&fj)}EJ9u~}PRo}VoEL%z~1o;b|{1OQ>IVXbpJW%jw@{JfBdg4|?+O_dq6+*XdZ@|A2g5R6-V&M6}RYhHM^IqMw2)?oQ74_UCFI<eDVe%8PL@R7<$DtI*|4anWy&(Qci2FYF7IY^oEcavlr8%GUiN+WDz0{CZE=27U?0B_g6fNfA?-|3Y#MT4Wz9|UzInWI`qVwg`^s62kfl%#4?hmjh9*X3!09|GPV-hcR~`;!;tN$p?eZGAl;`Tdo?`@WS39Z1Ofv|h7yw-_8c8&=@pZBNn%zU*`o!Ic-3Y^-J+vkWElnGV#RQhLH4!}Mt|+yEtQG|!=tO7{*)VWy9goM9?9LRo9M$pO<gZ9{rj;#tHG&%Wdg>~)J8wa;;+1Jr7*wLxjI=Gt;(*-M7sl{iAr`L-U;48BwQRrvpE$%&Ns7JjI9muIO_n466n9k~W{_q>c&9}he6;DyyDR%4e3ZVQE0(v2iZ;oxJ|enAWA%u{hMQ|?F>HQ>olANxet9-r9cVcs*UUfvi)>PL@9>ML8UAXv_7g_L_6JwC|aY1&gu<l-R4$Wn2_#r4XVdS;&HrYvL;dmr;I$UMA<9YQ#yl+#wLDX^!#k|pi!DSoV1wSS~HN$SiTT|xPd`nAO;2{a5!m<~-9dI5*ii0hQ8>AU276%=HLCcbtFm*{%m56a1k<<1av^)<y0qYwnhWla(c27<EBL^PxgaVw;~<cp;;J*i_7osBy02|RV0A~{-Ga5H0-Yxb$V>9V%0P&vB|g$vv=ff3F+N?~`5Rd{hkQb0!vNem!`qEPWtg5-iI+Eb(!%4RO~L+3U=!Kms_otLnPAOw2}!+Kq;*Xy|q3?V1@<=huIoV=JCc{?^*BNyt^UapI#p~t&Qxm+G4D|+-vJyRf{*lz_Qe3iFQ1IkOerpNlGsZxcb1_+I`j>JoF=tH`-03cH$4!tjwW}j|Hoi_Dl_%$6ZgPv-8$<x>Aw}|!A-tH^@(d@B;haDZv6r#N`65FKjlo583#w9s!0W_Y6QY5QIS24Ljc&%a6AQ<oJ^0vT;WguSrM*v26{1q&u<Dp*Xur@7!!bcKA77mIEL^Re5m4?<HW_CbQK~}ur42Kg}TPn;G91HY3J+RQGCW*ag<Wc45@nh2?e%264#zZ?v32%;6f}MHDC^oqn(rcEl0+;k7Q#|{)Y}z_G36P}@!kuVq!&y)bmjyHq%M;$+4%Ba=4N4HB4c!va9=28GntFSzTXezYgAgH$y1GbX2KD+w`T~f~Lc`xjV~*RL^D|VMPz_>`(AM&*yZi%mlP2tC(w=;h4XL&yp9#yATLm~a^1N*WEmKm@ZqxiZO!Wpa7(ks=@C&w}2Dkh!z~<MER!3V-gY4nqJJc&Z^VpdHO^wknEB0Dv&s6<28J_xDj|(IIWDC#w`busWQX^?w4fa_tJhqq-6e}bEkkXwdkS(jmQfBY!o~Ea05e@<gz5hVq*YHLCam0mWI5UKj%2T1pu46+u3A9n=C7u`-6Vp2!0$Q>&!ODOGn9vXph>b&$%ZtX{5Yhx0X|J6ZKC&4S1&J!`a79{R3Eh{#&gldcNt571_#>m5J9s)_is)C4hLG8HD~4(3H#$^odU-2TRf8lwG}=(9@qv9()=nQAS6b8XJ+|idJB-z<(vPR(l_%IGB&|PD#AWtoQsi!SjA{;o82hx4ZNP$zJs+@n1ZR5m9ICw{G3Ijm<#asP2}H9z1_KBQYArK_fNN2x-WyWqsfgH%BI|Hzwu;dB*iY0!G2UwOZx29`@zbws+$){LPLPg;fXXH(*5uW+k|;5ib5NZu<$|%3eOT7KL6%<bWn$Z&_P{WnLz?IHQQs9^ZKHL>2&BoK7ng?Jjh4*N*<-^<S3P;J(1~g2{t?;UNw!0#0{!B8I_@#6B(7BAPQ_U_UAky$Rgm$Jv;r$z^4Tenq>r^Tz#JDC^9d73l)=v)vm@aNmYMrWM4)k5B&so|QJ&W8=OWJI^q%u%^%-vMC;9@4QCMlFDN+X`6osemqaZF9m1D%IGio?05-k1fo{=8r6CJYQsAd+CD<pHujuq_Eg7}(#K|;oFBDa4sf<+s;!>~aj@yCEfA;oV4k1C14)mZ42KUcdOP~0W6Cu$`}5Z)J)`!u7fY(^(4#MB*VlI1&U>Lo`_U_wt&W5Kl|f}ehMOU5`#*#zSd>e(N{<Ju;pbP}A%8L>($jtu;pD9KAQYIj6N&R;eK5r#y=KvGJPxTGfcm>myZwD;s0(y?xD3oh;S$uE{F+3nX7SoTn$2n!{2(DtQrvMot)B-L|v7{Z1T0p<{5YON=Z9tM8Ut7A2}RBcM2=G}f?y%zGRc04qPYJX_83<nI9rqM(+QIY~2kq54gR)Zq_o-M4H4%XWW*w!na3Hi|(vczN^rNgVAn-Rn9ZOHJ1KKo8YjS_J4@+bkk4FkRT2SNfErUrua4W`lMISLZHEY61&j*b-iXjbX_jTX9^7>YT(?*7m%x<Hb3N*~u(D=1!H$m9&xm|**4D6o^t5m=RMRv6mwKsayV($jCOW!~TRG^2P~nssYIXNE?tHw+ZWZb+*B!8tkKi3+cvUGm4}IRA8^m7@lJ#r}4c1iV5-ku+!ejHcC|A5o#Qveitl10JM=c~EH+q;`3kFbo9}MSx>nd!kD{PPW|2#wxYxx;Y}xSfg?&R-%bcP6w-I>m>LzTaM{99$a=ORlM2;dBBwe^>9L-(Hmj+7D#*v0C=&K?diJd8*=vPdGoSrw^u{5vL`LwdY)SyjqeK-af^<79sygSu4GG#lV%zlbh5HRrLInYPc*%FH+v^1yFD%EZh&_Z8%EZx_vSN|lgW2?7J%flZD{sOKGQwyQ526*$<j<9E?b2zGpAgIp_MSBc?jji_TB+d@zI+MQqaqGG}N`aXbi6jKP8o>Y7nW2)`?YDY5j(jVkK!qZ=n;?cR&A`oDN60GO5B+0`<_-AxRim84&*{b>Xi%`Ov&EjHxxo6<}e|7u~svmJrkGKg$f=jQ*x`>@<?3NJyDKeG}EoE@mn3ZuHzpX%g|Hrc3ee{XaT$T&+is9h1fH%y}n*Yy0SR79F~LD!a+XG+!lT+c3S6WECG%{&p*y|8@EeKEaD0&iO<VN5knT0bLzYGY%Pf!Kp#*)CQ3jO&Z=;57)v_-d(8veg@;NFmE5+i}j0wiI$K|?5f&x>G3c&A6Kl;+HViGRA4VRKehTcHEoxv*ZcO6MK{7O=>?h$E@F+*hNzh}sKr_?66je0DNDx(xou<z3IxegJ9)8>U1Mi+7sta0hjEr!Vq7bsKj>RR=Q)#T-sdR*c}1T-q_0VxwRmabBSobFsV<BTkWG<5v!IFt>}vDGzpIpv08RrDre))7D@payE2gkpNc1WRja;XDW$|g8K(J;FwK7kT>;R=ICkWuV7V$lD%cAikMoN8B4TBvXn?bj65=aRlVC8#&u8m=fcv6yFSt2A))YEpusgXe~oEl-LtZNa7tjhjJGIEg}!1JW0fV2Iqj0KQH$dm<YKiyAR0BxofVSYgjuQ!*4S$$p<Jm^)$P6a19-_8ENVuC4J1=e%rs(@R2#9Df+&k)FYts~6qs5<)5D_;1g(UbVNC+}T3ZLW#vR;J#RB#s@tA4Lugk=qSKjb<&`nH<p)z?c<+-$&ZiUoywjQ3ie>s2-Bt+;91_6`JP}9uXQ!?HAmQ)Q8Pc%mhgUsmF6nCMzsy%$57k-j$A-$pjI+7A*(E8({n=<^qIuA*u@S4i$Fm<h#EWaxf91qJ2!@?-}o<FmMRWXjBNzwOP9XPG`biXiZ2RhfgDo-p25c8AdFRAJ{|IF9$UkBnZj*w6phn*M42MftOOwPTgrT-U~aQAxVQ~jM?leJPg_jrOBjIhsVr}+1tk2gm_zl)vd#bDU!ob^Xe`kHoIH{Aus`L;s%jc+VV3@QfV@!0IS;$>e5+ngL?uzg4eIM3TgXJ;;Jx#LD7shDUd(1pJymgW1wkyO;$Ep7--liEJv;8FZKVZY!R*TLV%+dk?3dmhHmsI`(<cSZ>OhMR|~d;Z^*kNF)dFSnx1I_biPtmwC+M-2DJN!=?&1VO!X!#7(T{H>3Q!aUSOHA)4WA$ojuF}Ld#fvwNkB}pP&e!RKpu}RFO_wQ+;i;_V9seN)0OW!yh;~BWj-<n1R+<TOKqyBj*S|zHG82^HARZ&LMhuULx)PQ#EGg9P+Y^b7<}!>c|;yU;>f+&k-IsDKXOz<aRVB2iwED(>;u|!44jOcv43IW>HwaH7x|S%@g>C<|3q>$r5xfz@iFfM6)P~TuLogOpozEaI@WDeaQ$>+m)wJH%XfIK()^eIE&?d9b%ZN{IIO4X49ziJm}Bki}PjpY$VhtPCIQnjAjDwgW;YtIH!jVr8FQF=}`Qmam?oMp~OtRMvQ{v9KePuB0-5`s91hflnArFoeWK&HHQ>9s*#OWw-w<-k`fzzQzlBJ?ArW|b$wg1{cgplq&_DUfV2@fCm_nDSjSKzcd(HJWH>v)_nBCg_*C2xq0P#1eVFP%Yf=#B`*G#)U!=TF%m_QwC4nT}zI3e6JR-t^Xmq%&ZHaHIL5o*?r|F#X8SI1j<YKfT99C#BojMl<_Yn_JLJwy`4JFy7)ke70U>h=>ya3^hS<jM5o!BOlOh;C5rF@2z#l!%`4(}I0*^wsg=!x)d>-gVgUDESMQ7H(xYUaN$ixlxPgC;J+ia}!-XdNO<$OT=(^^RJ%d&1S~q-Z5)nzg1eN)GE$ll=bV%~WD0(!}V^M$;`5da}mM%ag&;OmsA`uy1>o>NC8fTI~d?)k~MaNkvw7T_Z1w<5c`?(r9l0)vA@zNgMuD_SBN5w{>)CS%2SQhPg+{zxM?y_b{a9JOS-AF-pSIg3J%N3|f!{0D^p~-8w)__BJBvt3C@o4kfIqXhw;Un+FYm)@LQL7RPjJ!Lwe0?25b34qv;UqA`)YbS`p`J^~^{iXP2$0l7dW#KVb{6sLH}-lLi(4-2<LI=0xr7Ti6~OQN$fcpaj=Za`Gbh-eVwv~5Hvl0+|@h1fZI(;n+w<x2)9L355~`AY@c;JA5?W3&H4XC*u#=X22d9qD1tu{V({OuL#3s#8pkJ9Tf1p4kFW$JSdBC(`+-+#ft6X!5K}$iG=%DH{;+4<=U01ywbL5X3-_^Of_cLlFI6LhtwYgkU_QwG$-^!Kg&-a2rJyq9fEgVY@rNl%_OUQwhH<NU^F@6!W+|E2!H%xu-6UEvwXiL+YR7twIV;(k9yk(`KGfF;mDvS<+m>gzEQwtPOlM+B|%OIhpXsS!C(Xob{k#a$yTwF@+%BGN#t(x57-McD2MyC_Au|*BXxMN<@%->E7}8Xkw8=Q5(YRD^_pQ*98N`<Sts4a)`cRD&b&PyIS1S3J|JwkrdH+x`Sk}u}D7gJ|+l4Y-U}(U$r9%_e!wOVd!V@N&_jnp_*h72wLWhaccFQ<8-Z5jqtE&5uSos8dZrsorq54QfHNC{y{{QcND~b-$~k{Oao7|hd~_-i~?`}hnBjf_U-QT6D<z`OwHPN8-Ci?r}C$dKkG)Z_RN|u3IH|q_`DV#7!ASxE|J*?{MHITYx=jLhx*yVOvb~{`27l9TzVRP>}i=K?+i1IX}3czbU0aXqj~oe8Zv#<(V|LsT(mTjxRwsx12Wv&jICcP?D+IuiuiVjK4M&Zb0{^c#t3o{L5au^6p>e#PPvW3p;Z=?E;K|Jg28>O+b%v;aSv0b6|L5}kmj~aEsY%}*IHQEjtiC30A468(#s^ZjKAb&xal0Pq6fWk6mtdrT)kd!mt)h|>03q4o7jg)x$079*N`1)pMVBJrJ9<vq|4l+_&T+n&(djQ+(h0_pox)>!G+0rqr*+^NnSp)Q>~j)r6M^hT$)S8!waD4WvzdO(Gc4+$mb~>)iEHGVsZiMWU>-Jk;<wxmhYN`;ktB@GW>#)fn+q0cr(ssv06pX?kmBfyVCm!aH#eGGl(kdr=%7Wcay%M?U3gpbDh&=SOPObAm^9T)AUzCv5Oa<J)fbQchwXO>h>MYS&FfIB1E3=m@VzDK6E>97v?ueRr@I@Mwa6NHLf=k>a*QAr2v|HYn|Qg8)yP03!Rk%r@nT&D+!SifyxsFzW|=5ih2hLNWKmp$`+2gP;2HH;Jv|qkFUhWgxH8<Ac}<<&SYb0ja(#Ja<B9x+y<bE0@j&Ays6<%nz&b_;aOu{oi|qhmpK0`X^HLnGVsJSH3d|T{s#Jr6kGCyZqD?L15E;eHA5aH;8pJZscXo%T7U((n}U)B1YCXVw$CZY9F;vP8IvZ<1lyAK3~s4;T~+8N>9tH0^f_8f;B4K6oNT4Qin^(scF~JL@Ofh-h%tgz()nd|9LHIsgJp1KUDH;)*pj-`oDFj|GQ8KSD?n0)*g6=h;sBC8jo5k6@tIxU^H$O$%x&}Xok;`4f}qu^L*c2ts7Jkl=W)6Xwe8^-N%o@7(c9i#jp29Xv*nJsxz)sB2s_Atqa+fk4LlBF`1K3Qp@&IQ&SW+F5S)Qm)_YAhpJ~Jn6`wPrPR4M>R9B+_GFMNr=2*l^SmIj;ykm=nVeLo*Y{~>gc$7r8JX)ht%(oT+b2eYivi@vt_zetM8ItabcX{_qA*6{<=*q-HFM@)PkCW~Y61od<xo!C83%iAacJ5PAgMqrj|7E<G7!Ix&$<ZEmO<}D-PQ;c#uvff&Dreg7+5cCaC<N)ALoSbo5iG_~p}BTGC<Wm!IvJ1_c%SxkU+b^n{y-~>i+1JO$83lKWhrZno4~_R9;Tz*TqAdIyUX_4D~^qwhf>lx43<?AbFR5>67Fz&+YJO*g~a9qMBgwSZbI*sR+(Oyv<#IWQU<_E?X|FcO}G$>lU?4@ZI!1|UqL!8&EK|4E8w;fX`Lnm2}x!|!vB!}qHY8wqtTvMO0^WA%}qTs$wd6LG1A65{vM9Tnke`16gNEqX+TssD6D5Qc>Z~Ui8%w#o58lu0OK1$rJZuPKLsdsCeO;@cQMYo>P!J>T_Ic55=v4e0&{;^Ns}aLfvxp=T=Z?5Bsy@y;Cb6eySvRN<SgO6LkWJ04%caWz{B!x0~G7kZ__F!vuwri+k6s=nL{8j70zw1&FE2Iv)eR;<0Zz}80yzTea2Sg(P69U0&24BJwe`f$c8bh5nYw&Um|sc6@U=*zAm%A+D|F7!P1@Ld1{TfDB@{*XKSNZ=q6XnlIS++LbdOIs{Z0@^z1v?VA<^zld`Ox(kSwRCd94Ys!UEb-9L&`=Spq`0haJMUyX0Srt@SHVHHxQv0H@MB{`dfvhLYm4pb{+WG<@Im_0g^Mqho8uADu(F?d|sv0_+)P=`MDcrb_Dt^0?5mwGN&cFKV8L0Y+Hs8g12OOzcG^n$q|clAYImfnAut0ZV{J!J3fB*$LSm8a>#n<+=CLKkKL@XGQ7yE+4id;1U!Q=31nWMGzCxw-Iai8G+N$kJ_|liUCfnHVS!AduVBY)G?Mp)jI%N&LMWwA^K&WxBC0oXIbQ=u=PTOftYFTgAHo%o~7TZWLqcjm;F(fQH{wPi;nLAMS6ugm&L}lT)Fk_G*|;-wUujC7|3|f8kv%qUy&8z<uB)fA$Yi?@MQg%A`44M?Iugk1X_3H2hyQFTkAFb~y?LrR5pi+UZzWC}MpDKpxxNHP365=Lowc=<=<tzSQRILsuTwf<l^9*)iFoh`Z2D%G)PX1$`2R=L!a<WM16giQ!JB*ljzuxsMKBZBUpLhM$+#n@R0`4~m3ZqOeq3MoJB)|1;(Z#bftYk+h;l2-=~fQR!LH=7_7<Tbdc7<Q(D_4Oc|IRQW+k<E={!Q<3(R?NZyZ<-O{nTWZgAV4Awk>lzd)Pae3ITXGJw)w6H7c?4YJfS~RNpg5V9)e}*>FRcM;Eg=#`PHvv-kQHn!T_-s?QCMrOW?QMY7)BbhGfQfsg`$o3om&?zUGg`io)#_mOSOxoIUO9(QRt&nO<`qoo{ErIzZI<2t&slgTG@8E!dg==S*KdTlxX{!q9>v|drBc>y%L^{gV=Iot{}PdiNk7&`Wz2>`K^}pawAkG?@YVhyANSvaH@?$iE}xJUf*i?440H4EIpJ_Y*elB*yswh^q@*5N*?ncS%#cguN|GX3bPGU5>q>mdt<Y{lKx?@KeA;ED&^A@_Ddtyn(MWlOeL)=Ob9pa%c*q;(kakzky2=Nks%AL5Jx-x_U7Op4+8(Fa`s8%K0O`^uc?cVI09)fJ8_eC>2G&`lJ0RZOlo?Ja>roSmyQKM?#JoTcwGLE=)HMepFL8CC(IlcU9>`60ztKKZZAvX{_T})SxU^N!DzshPu^&8r9I%<ge)ikZ=I6?A1{^*T(lda<1VrO(f)aw2s2PvYa@XJ5v7T;Zb7}ri~Dp!4#M`fAWu{w%+MG6xGY-7PpW@_x2t_@bTL%4zMxnGMA-R_o1IV?FH=Ft+cC^D0AgO|MUlqkL_sFqTc!(%4d`3K2=B6Ger?KAC)qK^b5YdEMkwOYwSW&x@JN|I;Jr;#eXiIO5iV<S*D|7u+V)OlREmw`NiVpn0oJAP!)lE3vno?1Qb7kR>B?vgRAw~$$3o;?lNd_e4Cp({#tpQER0ZlOz8~3Tc``yxNjSNtFjssS$r+^f@XL9PxpRE*J5Uo>Q>$ifVnc?Oe*ym?s<RItmh}VR2)u-0z5Dd>CwWwfAVO+0LB*4zCezhr(qt032ZMI-de&>ap=ZxdGp{w^Y02PRN)O%FRxS+F_5GxYorzp!dS>N{y&4%R3Tqz4GowMtu|%H5&<&?7(QG*Bcax`DQnsLs=AQ4R1!38Aq<7Bnj1z;;Jkuaw3*NHer8ObMHt|aBdtoSo0<<U!sC_M%TN^o$4Y0lyJfPT$f@A0}uwaYwwBN~_C8(#6(e-kQXjq*tRAcjWlD2t#CPH|<E7E>(ZU~T=&iK3kM0360VLsQ>G0=Zt9Zfrh(Yg}O!|6N2gh2U$3>CVd-2VORzxee(|LbpmT3vtn^^bq~^X2}_{@=c=)_?u7y1e}Nub(m<^&ftD|KZ)2uiy0ZFXymzwMKJaGJE^u`ybwZ3*$c?C*CZlUw{7oaQG=SZJAfzemH#i_~V0QTOd{z73J=~X_Y;^D#V|)Lsfnp2##>(TOr)baQE<Q=wT$^_GD#7>ZjT{TF*e``P)AL-!>b!_4sec0b6}$MaNZ9=bf23C<%h%3ax0)#Qb5!2`edBuM*w>faf3n=<a4#b^SGpnBjjw2~v}9Yi~kxf~~S7SdLH!=<#p#9Rs$A&C;EH?)>KEJ^}f;q|RBCB$LJ$vngmROKRv|IVf36`8WAKQv97(Q-yplUPRF5D?}S5k?+C>gUqd2;<2AkH+;whbUeHExmMWtL?7FFmBDB-d6*h6kTIV+f3`Z;_sK`VaP};W7&8LS9vQTF<u5UgNclS!+5@D8vdKy+nhX5y)`$RK^!YdTcRt5$J-bcgzs{J1xL%h;l;9yyl2t>Zt77Y>GaOBlSi68i?W~xp#Oj^Jwa^66jEq2xAJub0-OA&2wpN2mqUN!$a)BHPmZK&A*q8b$z>@)r*OCWn1J?$FB|VmuRx;}2(omJ)W=|Ygzkb^IPm|yKk#5bvjs%TLAA*JFFnVOuCG_|bHCq18_rMHYJhIC!=qSrOlqY8`Awt%>-U+78hY@Vh7j0!(cpMW#ksdjQkKYKJXylO!3G_ywp^t45y^`qHqaM7B`}x}ju3kU(^<#g7AFBgceW6b54tDk@=Oz3Ua@yGd3XxpSOs+{fCpR`$vPJ_>=_^1Mw74q}16eg^{G_Vho5->kydzxS7YZ?vjeHJb6`R^66_*H0Qu!1E;{3#Px(Y!vUv!-zjH=tvuo-AO2{cA7o(qJA72QrrViXZvV;M`Kzzh^NicK(rh(+_kEluh*Y<T_HUXW-f_AW)nQ#(w2Tsbk;eGqfrdNl%JN+%S4JEUQ*hr>-_K=(lWyj=i#dP2m>lZ((}#yRd(eR0MxLKS6NDxPYjobs7-KgL+Cx^_OoR)lU)?cw!1xuKZpI<T(Um5Rj87}i=u^R|9{H_U4I7dt{!>ZzogB8;PI%^8(SoAq&h+`HE*)^pXw_6|_A>kMKxw}r>2P#jBQ1QM0VvL8-jo*^%l*Gkg)e6GZ}XUdBEhYkAre0=w&9N0Z3@7;EJ(xdZ2+G@s!7(`H+FSQ|#tn7+?mT7Z#c4Zz#1@l-5v?YxW6iQpfk`BixDt})~m{S)Q%<x?7qCyqK0wk!`Y3bzP=0P#|%ZEr^+bU2Xcl?`m5p>@IWAg2G)7EFbEp2^kEHg8q$`^$ho`G)VbWI4o3*;oT@OG&vYHzm2t~@4E9Dc1t-!1UYL#?ZJs3vl!v?N%rbG3c%23UgHctP7~rhw{spytC;a_ww^f?v09`QQ&Z+@L;7*k;4l-MmlRiyowRI&IC1IbRCgz`!$I+BVd-fcS*LAP0s%fq;D2XxbB5up2}>P4&id*34?3K==<jgJWh`Lw){Q#4vvYll+2sSSe#Hf{~_FGK+62!3Ky9ntE*0`fjZ_k>6%=`|SUfn3e21+8;_Q-X3nUmgLJ>S1>?*yPIe77}OGvCQ^4y!8ZW3B}W4zf=@wS0=cvQU+d{$y(fZjSh0~r?gg)bupq#$$)4(82^b+-qGBf=t$f5K&O0`@LEr01dMopr05+?{fg+RnS~U0C<X950s;Gp5E+emGWs!p2?Q2`%b8<kcj!)>=d?nS?KRs8cBv$~OkKp^V!_tPUXL&v8gHW!LVI+sbSDJCi*{z~5CFiJ#O@?-Ow6wB>fNPYFHnRub4~ehcC}@)S9#U5bP4g?xip(aqx1`Vk(jk@wVUnjAC@EGwfK|&}P^m{&7Zp^2pEC=xPIYC#BSm`mIGQC(i>?8<Ox<BHH~uL&F|?%ON}$X<V_5poit`cQL7n$POsdIuGc8W~JY*x2Y#9||jg*f`a-v$Nj4;U4u6qNKGMYm3BiQ31N!0j$$ss?y4(2yMnC_zCIbnN8wRFqU1x&g2QAhzx{*k}aM5~stfCOQroFK7Rzf$AWEdjeS>Z9TbCa{8X%Rs{urQdk5(92kE*#wCA>6D;~3d)=-*i8Cw<ZlvSg_DXo_bu!LLHJLaU-11n_3m9@MNQMOL<fTw*a{=J`UD(&jJ8CyY>mz&ohRaabxD4iSu$vkreh3Of&&ou<7ST6;f7VPB~lV&#Yk3yr8Db;Z4z$Aq*x;s&b5q7CuWgcm7hAq>xg4Lbe0Qa11-J}-$0TrsSw8f`KmZNOZZ4WHa3nX!>X;?kWrzqd6}ijoI$D&o~2VGNy+4`EyMS`C+-<X1tQj$Fs#>uHQ;vYSizEces__5iFCeF9T$txdUc#6@|nYZlFDltyc{K8mwH%&;(<jZuH?EE3YlIO*VFT8N`fq-*Wu3!;ZpLKxtW_l0aaYoozoT5xjt4bDIlktm+C-iuJ^T_rpuN4?Q(hSf|@x^5he5Y!~z(#+X47@wKCoT_$%9S!@on?o7}An4EM@-BWTnHiDU|{V7{_fw>klpttQQzx#=V9sm7zdMwO0=Su{%0;enuN50$KtXRwdik*iyS27rzd7^;SMNl<J)GaZ7VI(5lKIWgm=4v?|Xn>UxU4^8F=nsz5%cw~%VcKjcZ*FGXgp3h~y4FsxIu{+FaXSAEAqf?u;XwNX7!sV^rLd3~L6jFtJEwEwa$f1+Pr_v$m$u^)$y6sW;LdzUkZ>BM&u71eNL?I1m+Gr7Oa%xJS2o0cPA?Vh6#&S~AL;<(RC_^s60bP!SkiTx!OGdEs*PALj!XcBFf|Nv#!d6neI0fPny{GNw;Kq6hfBt$Za;-s<C*UVio}V>gPgkm<FVyDKRS>1f<p_S=#!Azf>RrCdu~S=B7}++RhK$_ZSZys7USQn?G*mF8^}N!UeB5ZxN$A58)5DYIY0e8vspvpxl3k<gGJ&p^;KN!bD9Ig&X$(0c0Cz;XER=Mz@Bxvwfw9PF(!mZ6Cv<ck957;b(qj(4Ccg<6`>@QRYVYvlJR1M{*;mQdI^o#S-Krid!p%w=LzLmAZbwmS9#otzq&X>cnEbqRB3$o*s@@<B^=M5-p*2*;Av2y*>3v`$=AC9SR&1|1;rekrRFJGofQ#WTI{UP;{waHc&bfi(lH1i3XIFn`d<!7vErAza!c(619C?6H=MUByu|c3mz(*3x8vq22#Clv*XnEKN;6G43m<Z!dci_5Pj`j3t)|*}Gv=fH#=7WdrE$suk?F>m_k))VwnPKQ}#r9A<ZpKiYCM-SFa_nrB`nod^lw%GRc9l{dp#fg2DwtrPYVCG+TcOI=+ME#4AAOJ`G{EE;SfiGzDq!dplAaBKwQ!(8R^e>x<TVvP-om4FY!R-C(B!EzIYdYbt#rG5lps(ksYs1Mc`wg6x*%Aj3EZK+ES+^?i7(~~;@|QcGDkXwVscD?B1zaITRtpiB;Jpmfs%+qAY6C`Ok91VsI_Z$O!$S?r4Ka`Om5hg93zG%^6WJ`^)YwNKQFOs5;~q5K?C5=azd_9Cq-W_;bA3Di9LpcL6;9Zh3`#prnul2z2?XoO-lc3|B!U4XRKD3VRF{A4akjG&nB@k0&sXTq6v$?IrIQJh8>v$pJUSI4E-t;ZO+!<CnTHe)uxAyN0}lAr8<~BBOVmhZx^NPL<PL)#IzWVM3QV_ccI#{)1DGl?t+FV1!p7LzsDdEon*tgH=X35w8c<e*Y+h@^hC2*(Cv9;3(Q{Ne*Ce$9h6|JqAd@|#*Ap4dAs7!!FahSTAE3-XWZan^{3M&Sq@^*kkc?XV}nBW4h-!<v2iNhRe=;mU`4&`rG?Y-WeI;$IaG%+#u4J%gPyukOC}b=l`LH9TBx}-i9@1e6P>GzL?h;Mhca<AE(uaF7St8WReu-2Of8yBVec9T!6&4MxPtBqqNruQd;gCeL;_<hH5$Q9lhsGWg6;2(GEm6M0$@IC-tSD5oSGu+4jpSP+=kJnL`L^~3I#_wEz!j-^kR5n&x@TpaXGR-A$Fa1Gj9@DA?ZttLwk?=|I9i*h9)<<X09!?QS$~wMj=)QPwmFh==FSSYW~=Yl7x0l6SasgRR~q@wX((Fdmous*#h_J%7|Htpo~(JS_Qm*F-fl^NiyFmL6Hr5ptR&5-;t8<qCGapI$0l1jxB1r3ar+%6u<Jtv}c5N6yV#ZLTF0*qnXtxmFNMiNu{GC$;1!K<Tp_^v%`7MOV4jm9Q7g!ln8nhYj_<E@T@qg_9TP*LjyLK46Rlmm7f#6!4b;$3ME>{ly!|cfxymt!iy4V(pgF;tmjMO+UXpXybqqqk^|;lTNau;AZctRDa$EhMeVa$e^=8-flh;G73jX5vkV_Kb4R)>>DBYK!1jhJf7=6MNqjIt%r>;pPnn=c`1rAHGYHK#)C9y9QCw{tuCCm<qWVx!Uj-FGSvw8&v}M*^dDc->7&kfdR=bwxBTwWS^EvFCLq8oON-M>LoA-{C)CUXgTsmouH)l(<<UaAvwex;}>2JWFK6pLdg~t<^&DJ#sP7n-<J4=g}gSKfeuz`8G2wyPDd>SIfi)M2z96I*mWoQT%Ck_9c@$dV#>{`H8M1Y4s-n4_rI_Qi!L9V)@&!;gc#LFark|XD*G%XG5=5NfG5GSZ_l4H=Wv(Jvd^jl7~+JZOg^v|YxkJnw`7}}c`L*F+7YOc4Kf;N?wZ_>^N5_YdxT-OY!G6$4fo2dNjF*dP6@1zcN7w5;Wb`M@!S<bHGOQ9!COy8+uF)2koImnDW;MMOGa-c9IP2C4FwUH+c(~aWg?z<^j;IoGCqX7B5^v0T^osLA%y%K4i??XWD{4<!+^WM##TI57yeN_7G+2sb|AgqPX&Ijfy<maZ1;{lByJY8}axQ%r07Swb$Y;p;(Elk}cd@_1eARQD=D>s4&y<J&G4F;^j$Q3UR1>>Zogh!vn0{y^}<!|>d3NfL*Dv@#(T|k5#gX^W}?nMu34X*7tWAmXTjTt<WZ|8-#g=~88YQ9*S7w>@-V%1InBC4hb``8d9Z&lw0z``rvBT8*xwT`{3o`dbdc%DKi58KZ-Rid8Pe14R<9spuv32|p`i;A9QRN-u~0cQ4c@?a2-DPm9)dPHcjH4la1L+<n$j)#G2mTJViPP2Ab<0vx)fX|+ceW95Kc4p^5Wea3ns^ViG@n55k5@owqgimo&OY?!c#oXF)_G9diBtB2XmVL^SMayW9dFXGqpU>(%R@O_|ng&50X;sA57-vS+CsYOYnOs>7js}OOWOhu8*>QZH$K88Jca2^j;e~#F{xY4mf>Gg>;pD(Y!Pw)L(Q24?;ULFC$;lx$R1JQ|A8%$S-c6tGs7t>}eQ2x01_?6pngCcy2D6CVC$++_0W2Cxflw!tq)uxijc)KoFAk%!<NU@TvEfcG+^dtN05w6Ah>XVR>GTcq*tlw?@Q)^s_tJ9QG*HY&Tp%hf<MVC~o+Pq-1cLqE4s-UrYF6-uQ8HSP>s4bhgtU^L%-lq2IULJ_QVjRRT8FXS$W&xRIBFH_l{HAUqCGnTMdqji6uP;puFEHgx86TkDxFX?Vbft4p5*0RA~414T8|UeB#3EReHN}GH2ynmN#c}^6Zj9Ny7{)$)<gv;(Az#cc9mN=LXeU4JMjr64eT*%W9b$r@Xfi93h%Hjk6dbVqUMJ^M~PdBXgc$(%PGl>W66w6{LxQxu5U|<AMFUwkBVR$flKHj>rVt;2$)2pKUH>-q~31-YfU;|JyP&^@#KoNWt97|HR(>8=QC=Flu!p+L^fqZ^~I2Uq|lTfqy!&F-2O%}LC&YL71LaH0woPgZ}YU}k;CLoTp74@UijkEfgL>t$`pQ60&jLAfrm++GN~gLWNN)ug;ziT*zBB2qhuN@78f}uYc$5n%~gmH_yoMuSdAlO&jI+Q!iP5Ac88d)*8#Y8)*N+SV1Pq>MculcO*E##69@-o33WlKCO19VEn{nfxe)_qcuZ-yrI%_!YsgX);UVsG_Fq-KbmWuhG{Kn`DJ^2As`8O4H-2mfsgR@-n>S*b4iiZr5)IaJPjO&zH4^0{lSRB!LR|5}qS%BpWgBpFTM(-)2}D`;4BAm&8T2JZ2$vCEre#D<5TGwOx+M`E|0ZJ3+!oYtZFOwNtb`f!;f-2X=73+_9OctW;nJ14&`K4qAh61jRd5W^B{53*6j9+LFTeIURm!Kwf|3c9mxx4*n5T|m?$HG|us^lufxq2yBW`wJu8k)zjnL^^0k%j?GSPi_3?-eM<hnQCq&)@{Ado?O8dqOJhZu6kfQM;z9Bp#d=U6ob)I<*G8!A7>(H^eLa34{pqcZOYfq@n;STRW28}D9G!@R}cMGB>gT3!V=aXBDt-@sa{5E{D-XplObq{Hqn&^Bl}PMX9-OY-EWC8fb>VEjo`m08Yz2(X%zl$r_jl>~j*b8Op(kU8ygpp#9v`VsQht0m5!q4WfxhvD0`>1!?04yj*Hk`+0y%<V-Feqwqto&Oi%P$@korDul%#*y#_!*7h<CX$IF@KAx>B1%I<-*TQ6oYf;d7<Qc=UnmMz1#c4vssZkw)Hd8!R#Vhb=J%my97Q^RZKLk!n59N^Qjb{=<vs5%9pr_TawiRcF2=VD4sWH0Qe7hMq^30-mP-tPoNAm0PD$sy&q|@EJt2O-a{}|`!q`^4$!PS1ahDbQHF_FI9E$2@DvUy)(cC1QFKm?yNX8@$4oujP<MK52xIinrSz4R6ZO^^VGefskJc;xKw|1UEjKOb0v}XU?<SYHAe9m}L=NZh+5*-x1z3kci&7wb$xnaxn`ejwi;X>PDvN*ZfcIwXB{&bNLC6VZqHj<zTvk%bMY*LP~!C+KL5EmUmGie?qp`4*|lj{k4+}~((y^g|M?Do*LxoS0MN2)o9D^jgGGW0CIltEfb*-brG7dOM35-n><9L1@LUTjk+>}#c^k@EzmWGW_2lRliyos2nLl@2v>N_CZLbWCYAL2M6__jgs3;i9^el1@8N=3n@JW>p%8M;;j20wT=aeWSolNgyGsZIeCUIWH)?X9%iDPwxFLy7yI+tjVX^@8QI?h2vRC!8Mc$H=J_iJ-u`kSlOO328?KmZfxYHwr%-KlD2X3cJ+k2Cq{->eXm)nyhX`93E?n-2cX+UT~w=ALMxAEl?_*D-<_tOMFqd36x8J#7?*Gs-8<dtK|@v5JLq%Mq&>4y%;`&bebai?CHypn>i9cv(F~u6n@Z+5AK^CqCA>xYKj*QVi=8Td29nBc%(05~t!lnjk`#TW^bP@6t>4QiS%OTX&?(x%?&63}tpE<TSjQMBGGFMN?Kw7_?2yTY>HRoH!*5aUg?X$|_)%`A6=+IUr&HRT+sg6~5qLO9!}E~QJr3gIkowlu%N&FiVn9b=NFt$HbM4A$d6E)NzDg9ei<m)TLtI{b-a6M4i3K-Lo<wyjasXuUtu+CbD<F2J$hS!=2{Dk9R(lljXrljZ`5mQ4OXauLyN(oP>_n%R3&vqoDfgcA-Aal*j@hxN;I<@1WbfIa-lJqIC5kD=@de6wu-mm=tS0TEQDh3nWDwgjhm}HQ@ol2o4ir0TPxC`V-C*PsT*E{GQNC8RiPLLIo!PpKyoYMTh-4lSJxsY=`r?RL0(E6Os{DgNm|!>O?><GjYJjq9?1lMTr2C=r+^TCb%gUBHisp-O0%ZwoLkJdXm0d5RRi+zZdKd<r!)gh~%zisdX?J?bkTroaU8Tt~aG)8LM6nlYxK%x~<RNgB=)ooS$<3%XOxnOKGj4)5$3CBmO<!T&l2NnhCyZ_b3#Q*#8FT4zO=UURV^3x6U<)ptQ%bILAelD5G%?6rTDaZTZAF}oH~Z!I!DPz&g1uy`nUNMC3rcO-*$oomlO{L|Uz1j=p+aVeyQwrOC7FuR=5{ksLyk&DX34tP=@vf%tK49ve#v<{IVfn$%<ucyhoYEDD5|AK_%I-oaNU}~jpV~b(V&a8H7`J5Dv0itak;V<C03qEG8MB|H&w-I_wK6B!?Id2v#n)MO63B3phc<>r}T)aUJ2s<3f<d%^zK9wjaZjz{l>HjmG!0C0Mv~YV{eM=Ny+O&-qeQ(roH-4&7Txo?&)@F+<o=bKBY9=8T59QFz|d`--U6>H9{NBUoOSTyK$5v#_$RmJ#SAb>}d8~{3_US3;B21lOko2z_*dYy)&Ua5jm<*8p?R84DQ&>H^DEl3%w%3TUF|yN~f-MbM^qN;DQWBlsVX1V+2&1WZFfLkPX90YB1#Jg$Pt#o+^0;NKO-djDZxHUa{{#QES8Xr?xHKJ973wyd4&{NeElIQr<D#TAe=VX--L|IRi9m8yVBppU{SAuWWWG7u>_^zUl>xOe8&Qx|um?v<(sJfWS&&Hz>(6;3Yr4DDkC?3o(;2M7W@N!n2DKmmOtY4B+YzJyFit(Qk6vV_MSIq9q3r24Nd4K>ZtQfk{7HPLWlNKD?m{&x4MY<z2lIzuOW#q|CFdGc%`^QA3$;SA>sNL-)wJmYfgRQE94lc}5tbk6&l}>EmpqGY?!>sxE)Ps?`GJ9`cTtm~cv{i8Uhr+-3BBWiM7sA#0$k^{l7z)lIaQmIscpO3QnC7ty3zENFASJcH1DwO_1BC{9$hl=Ti89}W`f5MD7&_a*D5!SYq`AgfDZvGWUFX{jlR=A%lsbk$*5SS9Ld!3F7j;yV08j3s5Noy7_j?yFnyU65^eaJDI1hvJ^?D?B<??-*Vw(IItA2aiISO9upW^H`bImt<<IpbfXAUW@lkLwQ8ck^*}JYXt(;>ybSUd$wxgTvonObZI36M9y=vRa8e&>i9W|eJYtcL!>o39BI{(HqAwq+GF6B#z46cgB~i9ep81SX>>(7DH27iuKDXd53eh5nb2XXmJ1r_MlOQ%pyy!o5Td5coHGsS@r*vNIqKI{8Se`TawgJMlwt_N8`4Q%FE3x4Z+JNCt>M7RJb<P6lq!&vR143f+=f22Vlg5)GddCp<x}H?NMFC<8R6uU`(9Kwx_F<kMj7i%M;H6dSbKEv1gc!DWLNknan-9rUEdBC%HxW7oM|qUzl8UB4^dC$8quDeAqqF_oSo&<l*GsLgB*7j%ii0T=G&K2aQC%LSFZ|p7N}FoJjoGlZwQs)LoY5-?QIFu5M92Efz-Mx)p0b4bX_98u^x>U=hLe^sz+^I)|$Cy6fc!7^p```Ilbnxmi#ceNL{y6|4YZrZEXY}R2YnVLq_(hiS~FUpDC(Ffs29)9@J9Q7bGZ(MXZd6JUDcTk7p)8rx>xMabJyUkqJ9xTEePaZ_z<rsS<X`;Ymdsz4h0{!#S>Vu)t(ZzRM4Zc0*~6)EAT$E9lwnAb4&}0eNBCkks1<GH_ElV##(q^joB^O4q94oTZw~3OiEh@$8km<KqCBm+F1;u&V|qZrw>U<E_;^d}zCC8Q1*$Md01Bd8s;qbOER&`i(H?mtC10VcCmgAHn-e!mZ!BeBVTbFq*K1?STjR^-J{oEnr(wHHP|Z2Awj7)2bMKNoZ*@K+E=5!V>8@fX_>Y(^MAI<eBAMPN^<kcNF%;Vq6njLDB2z(ehl2*0Atn;nA9vuz_kGMMbDGqjo-_iw+&H_MXr*Bj*%H(y5kkSJA$jCA~-f%oVIcoIX^YtY$2&SFkRSAGIobTS8*&x^Ho4?*MgJg7H;aDq&9N>KwV&B2Gz(SqL-EfP!_5CteClV+N?jYhZ<LL`*0Yr^5<{fgcGW{Y@--sR7Nt<he`}DkdoxL6lRo_<-9+a?c!x>9gF0L0-Zi<*73N>8Rm>(;43MbxCW=m`sqoMJ*U6YCYLRuYrI?D(Tn}koRcmGG5goi7ZD@1@0un+`=P!ZJa0YQt|ZL5-n_xfbE?q{9c|vwi(KgZh*$Vp$cb9phag-$t~Aki>VBEkkj;UA8<#&&bzLlOn`_x8f1EoR@1Bv@TfgN;Xp)PGC9~&s-)EX5Wa0KgSdL<Wts}9s`|Z_4jL^USKK2P&YBMc*OCJH0j*3|oJMIsz)@w5*Q)pV`<U|)o%Ura*b3iZ1xP}Y1-#JoM!f^&+PaF??x~KD@!-Jyy2&@H9yZ>X@OP=Zrz!5UU}$9;m(fcqX_PQ6&C2LxReaWZS}NLkqf<5S6Y_{B?rEt%$>pd7D6}+*w3_n-3VHkabKY7aXN2EA&09g@jd~hfmc8Y_J$+PVi0vlNKJEgJdcH1jxqAkNCFq*((!YJwz>cL+ExumTMoL?I`e8=2c#h31{9p}ij(t;vW*gM_H`v!ORN;&0cxS1vn>{AaLZ;$l=C2iXU8w+SS+J+jYFfi5)wcB8k;tj!c*05zN<gobnUWj=JdE-z3``kY+B|$Dk6q?ZAeR4;M;m*tLNkbFbnY2bUNi^^jGs~hFG`s%TzLI!*(#GybRWsq5`$Q7$hRU|X|s1@zw-Dx%iql`-+lV{Q_n2JKN4Ar5dwPuX|T9@KW4f(Vd|7t0M;PjNe4%E2fQW@_%~#m*m8h5OY4zS<l1j-m;Uq{^1o)k@s7rM2b;Pf!{hJUzkmI&U;p#J{`RNU&6i*Q_?JIl?!WB*?aONQ*DtHf%YXm+7t;a#;g|Ox-hKJ{DL?;m4i$E`+rRzs{SR-y1=t^t6QcX+*Pp*X9DWM>DCULP9}XWr{`gR|f|ZGVSX{gRrWFnxb9D?$@0jtc9|wYQr}<V0w=LW~IK$IOzU|2>s{yqI<U6tlT2C~ehkpRRZ8mQ2<<^V?_NUA$hAT!K@M8aiBsyYHtCd!;=5V%3poEW>6rNWJy#&;f9{%X=W{NolgXXk$wu7G|qRjy}!Ku@=6&|o0he#~@@o)4U0|e4$>CQfPe)D+SxpppD0uIt{#<dQ}DGyW7R+dy?y__pqOZhkXJ~AuNM5ijfUStso6wsBdvOrG%U=S2FOFZ`T>4p!PfP(e*d~Q!FwbaMX-YT0QnoJ(1#tUT3r_P_P&h>rr5isyP3nRvi05S+hFJAdej3ZM1j)nH%;jKOM1><#+#zYLh%6GR$1o)!Qzp=kSVXd0o>iuiAn1u8Pfln?vDJaRR&%Cq3<UScFYXNDKY#NZVR3%msEv`k(B`szIV*IF{6OlJ#U<s^$C{{h`yg+^QN5RgidY%N}Dg8Ju{!*&<vAS1&`CH&5XGDzpxHMEHxY-j2)~}y7{?p|5KG#gWSST<!iao_GjJ^u>Vk(bAO%l%oGjw@9Mu^c@wPbKTEF^0cejCKC_)g&MB|X6gebH8yg~u@=6zNr$`S^{niAEl&0$gtdD#>FS=#?b7HR{1Thq8V@z}4%=zJBa)@MCoVt1r|E3hwhiIWOU-kkifvP=b<tpEI?HuN1RJ15fEIK+3vFeGmg#HD~;!s@|K(vKM@3Q*7TG|Jc>vn3yo}7JFlfpvOajVnFm&T9utHAcZeF=@%2K+aRcd{iT8-Z;P*B!g$*vsxLYx_6FKDItIfeNost{N<U3YlX?vsUO%=MB-)7$Nl{PH4ig_&PK<RQ#GG?{M<7g+C?NcHNW)wYhnvEH?t%Duy8!g`gox9K7JZ1W%y6gbi!+81swmS^@l@m2md{*tOr%0(1p!kgS6c$r9$vqb8;Y5(1M8YysYq-kY^_B!Z|m20!>opX=%>4ZR_du%SB5~$c0R6x2`<LHd#z$US50j107bjbAZAn1e-?^kNsK_E5-R&qP0U4+XX{y-^l+a`-R+r9-O0k>>+|v5n{r^6se6^Bs)I-8g|yX-4>5?KGGFTGtsrZ1pYMu&mK7bc@X9<Cj&}mp2ec)P4irjT#F7rJ7^c3jCCsUd3#R8TeN?DO8=)7frIJDxxOq@~5=;6FOw-d*OlHcp_gw_tx4@V@^L+&E8G2jV+EH6)7WDa|NMdlzcu}W!QQ7fDqo}>v8oTnCNOAbJ5?$rNI}f$4+M$}rozjwE-yfvXGl+UjGv&lS!m*M@A8WFy?F)Pqft+m_Y{1RCt!wO8@VT_DuEf`3cvohn%jH{pxZa|HbcKhd_<snIP?byAvg_2N1ph%_>6jiT?LAv?pvVO<yP1Yz5tg~N^uEi%%eN_a_H~v^UV!2MtD?yyYN@Ox@h2x8vdWUCU#+#5<WX8J#b7o;&8%ifud%)?qLpNf0th;bBS|xZBsOL+F<IY)-1=-mjUbHFo1uGk%(N)nj;dR2gi9P$Z*GI8&Xa0TW{S|hV&R{$%I-v=Xwk%ClLIBO489dKxpcFV$wG?cwy#QstjN)pdh)FU&y_q%|Mc`J%dP-8AHnx!houev&XQI_@>i~sKtBjw7P25+&hHfYBsq1X++p?bXn9u&-qrXDZEowkQ;x#j+Nd4*ixKxYDrluC^q0=*)aJWMJXi*V8H^|@s>SZWBomCD<ejhz)WdUBu+_|0fqMa_dkRRB#GG;#nv$hO*8p55b9dYK7WB}gQfisiQNx(6KpSAHl4??`JlFgt6uGnp_$a*=e>YP?rO!h)GRYBCVU0-1l_ch>wSfqOeDwOX1|lUgh0Z>($33}IFY#v~w}7Q-zpsONy1{f;#Ln5sJ1P`g?#pMgtdA-ISkI39ou<gNggGGyBjp5%z50~`Ud6PokWrKRmqmGLa-mI!jfdJKSX4L~+>Z#bYUacaGv^96ll~j|n*><lWW$P1Ad0XnAPE17vnytHv3hGNHL~S-wju?UUmreliv;iBW0ddTQV}|nbe@Rw)g}2QFYv05reh3Of&&mYn`XYD;f7VPB~rFxr9f8fq%-S-Z4%1Dq#z@fp|y-lXC0ATm7h99d{Erd=9y0|O>J{Ll59!EE>`xc<LE5mBl+0aIGPNrwrWF0#axlbE=^|3Qho3&of=6p9dB)ox#vA`&p0Y(EcA@SdOcVJZl{hFEScwb7ulCc=PT85u?VeK$4MeDc&W2gUd!O+DEYe7pb-=gEGlg!*R6EM=w)#|J&&d&XbyVq>#PtiC4acASnjZ=a)Q}+NqAJ8i`RA?)IGD!OGS<{r}Lhjrpp!i!-|rW6{&8TA~H+d#F(BmVL|wJwKCoT_$w8M`9N5bTTww@WPQsuYJ<cN1y?X%*{fTfM9Nl^=FOzREovCXqrFCz4t)aX+wm=C#90Dcm1eMy*^#STg9d<(5*Vt6cS%reJ~JJHp(1R_ML99!ruK%hiWz38P5*6mexM_DV)RDFi0rNDKOnDtM2<Y4<a!%TGPXWh?TlXPbaZMD6K#maQ@Fg<+iy6Th(fB6uLU-&=g3{&j?=4T8&D<P_Q+SEWsZC$)0k41_Ty!ukOnkumIyaFT1mBw9vurol+|-hlNuumxJ5=8atRLTawLTObtBC&f}OwKRM8O*nY<LFB(lr3lH$cF5Rd3R?WG1c)=T*F*He*e4U#+oKaukM;NQZPs)*;b`E(UTDRMc2U$?Q+bf&sDvU2RyRux9JO{XCvcQ;mB3xyY0cL5C*4Ba@dG$tQ6nsXBRu*CH6bZ46L!cr<a5SnDy=(<dOswMcamI+F7Z=?S)M+D%GNOXjfP8L2O(l#&_8BIFa!Qq6C&VvI+%uagD;n%b!;bI?_IaKW(ew;^R&OZAp**7K}JGxudV@0@GNn?mIywu7kO3j0c(}grAg$|RScTR-sJy6vfgrOd-$td(a>NsS^Q!2d=Y{b0NEXIoMH78s@j)w|*Yzc5N{6%M<R@OgdPtZ9xa9nb`nqph(?~Lzn!;BN~!b^C{)1D&_5bFHFIwLj+^a%J!N^k>!ppjUQiwZ3d`vCk0O5=i9{$beD9k}k+8a+Ll^=6kk?Svt``QTxDOZ$LsJ3~@fBq`=vP7nH9u{{)zn=urp2}=*P96KAOzU~YJ<(NZ-U8R&qXn@zs?j;zgTD#rdR;co|HYY^%M<3(}4KR5I)~Kbb3K$w(q-O(QEgWc&btv09c}*{mx9}((TZF43G<oVw4iS<<E8Q+1B?weXDpF%m-pezNE(jKB0(Yn{OJ|)};)}V0__zFq%#n_vm>d(JND}tQa}J9ciT7h?pd_LY2p66K6Ib6TYVDdG6Mms}`9e+Vle>{6$B3bcJbTSfeav0+&r56xgpQ|1&;a<eoRBNjNzrslcv#6(Vvpfq(B;EU;d>LDDK7X$uk)`)lhXg%KO}>3h1F_G_50Se4ak32&nB@k0&sXTq6v$?IrIQJh8>v$pJUSI4E-t;ZO+!<CnTHe6>o=)N0}lAr8<~BBOVmh?}eo7L<PL)#IzWVM3QV_ccI#{)1DGl?t+FV1&apSzsDdEon*tgH=X35w8c<e*Y+h@w3x71(Cv9;3(Q{Ne*Ce$9h6|JqAd@|#*Ap48JFTf!FahSTAE3-XWZan^{3M&Sq@^*kkfGPVuM1K3JmQ*v2iNhRe=;mU`4&`rG?Y-WeI;$IaK<Z*#^N{+uisj6AR%=7A|!y)clIXA<?mkw#-GM5wmbZnK&Ak1SuE`>I&tmzYAcd7EPwGca4MK6H-K6fsD1Fmig}eKXwoajIq?x{hKDMj{tz}?~O80$jSm>K5O3ZOq86OBJ2(w>zmt#(WXR3_k0QkM>#Fg#Vzz=cwx_rojP$jvOghqZDccV5?CSWON&E$kNf}3IzEObH@ar7EwoYd)k8)hRtHb*#?k2Yd~0gv&x(?Sc1#nsh%QwKRqwU3#o&7%nOE5Y_vy-rS&E>HQj=N*ynZoBuOvw_-zq_oyWHZL!664yMtMJTtdsTO<k+H?tH5eKOYtjTOnXLXM*+TlDukw_Kbl#MQi&eGnp8SUl1%)tOnwt(GdrC3y!8AA#ZfP!K#8D7v4+>t0MCk(YELq_KQv%-$<S&AQu#U28yum0uTY|OOj*~M6A0|QC%h<;CY_~p!g{_WuAR<N$@}1$EIDA_wPm5n1Cqv8lCqp4R@6S5^>;OW6zDW~R)Ox@Im_@-Gk2uBl3qPc3ye;IFudCy5KH2N31YUPg?`EeJ;KM2ZJR-8wxK2<wus_t<8XE5&K1>%g8C|`2+G=NsHZKn?#i={qQbbznYY@tJcoB8*O<>?=N$U!7*Sd&CfvMtL_RivY`7{}L}!<QH)l(<<UaAvwex;}>2JWFK6pLdg~t<^&DJ#sP7n-<J4=g}gSKfeuz`8G2wyPDd>SIfi)M2z96I*mWoQT%Ck_9c@$dV#>{`H8M1Y4s-n4_rI_Qi!L9V)@&!;gc#LFark|XD*G%XG5=5NfG5GSZ_l4H=Wv(FB{bSmU{-Q-%UEqK#jcY$MQZ(a<2-w3F=-eL;cR9e1CI~z#Yy<%})GoZ>GP;PCa@~_9(#0tHWI?!F5AG_K;cxh!hyN)k~o-{Fir;5d-6!qjFGxC5}zf;J8!jLp|AIQ{3o-|B1ikG|ZreuN78p4kP<nz)SYl?O{5<&M$q;<Xz0lD+fU`o$>H+yQ46OHvz>9=Q>8-#<f7CJj0n5&SVn>LOIG=A`O$zk9&(!E<y)7h}eCBU{Yb(8SP=uv@mP&lpJ2qN@$Wf?UXunHqryf_q$ladl1eHIJ!151{_-M=Wrg!ZaL%2jj$5q1o&m!i8DJ*YLfw&RS=hmtgA@JPO$7v2`K>A|b{VrgEy2U3VtI{}EOnjY-2=RsOPoWKY(%kfvf2T5&UwT`{3o`dbdc%DKi58KZ-Rid8Pe14R<9spuv32|p`i;A9QRN-u~0cQ4c@?a2-DPm9)dPHcjH4la1L+<n$j)#G2mTJViPP2Ab<0vx)fX|+ceW95Kc4p^5Wea3ns^ViG@n55k5@owqgimo&OY;HGVvY`;ae*28BZ<!wv1OmKWYIF(V;=h3?dP*PkCpXOwx&S<A}!ZN4uR5yGo$Jgssj5=uB-+}gF{m?JEq0#I6lwg?!BYCMz4?XLO(x$nNC~5sPM{ga^Rw1>~YIzHO#xv<1SAAUBSsAHdGCM#~*KIC*Dn;?x;(@N_}Xn!v+a5@tOcwNd~is+$XiduK_F?Nr6x&lcY{-BaLqGMK2Dcvg7>5AhF?2F5Ih=r2sWSlZcGQ>FM+h3)#48rSOj?k9Ti5CWAO%pc`?4sI-jFyE%B0$np^g_IrCfUX6jXG`8S9bM>mRNFlAHCo?xuS`NqZpcKPBvDRU1H!>9&5sq2~du0t$t!U4VK#@7B0EKR@s_XLQQmqdUhe{_DP1tl8h9`MBmk3Ppy4K@FH3?#xR-c9I2#x;^Taq|s;{^VLscyb4wKY)z3iP(mj$P#zju2!d{Z4#BNdtS#+E}{934C)dq{2IF%OjWCoT&L>&r#x5BAU)T>vBpm<5)5y6MyuRoa@_?;zv8e^P?izM&J^<$odoUN&+U)=ueehB&oOC|5}sISC14tUOc&CZ5ic$Y)!h8=J||TA|=#;7LiTaP<=5ZA1O5D2PwhF5x2ilOpx=bY{fK}oj^&$(%U?3dE_v86ITW<ofp3NbYMr1fii{Pl(4LwNZ?_Tr%dXI1({l}RpAv705&_P(kPk6ip53F$r_EZa&r|T1U|v@G*;sX+4H+;wTPkuCcAjs9g4PI2Y}mI6V!Qu0S@sMb?bIE(U=BLARLq>)CHxQ-1KC(jI9agMhuwYF{R;_UaAGHAxlk!hq%w#e^vF;kx!!21ZP^Lw1}Ci%15f)_^}<NLXu8w-iT>BOeBFwG+4_$#ev1uNR*RI7V%CAam5RZViV4kZNSZKL9DhU5M|jjXh(fz(3cb;Tt;-665}~RfWF}9mNeDNrtKlwJVpK1R>yYCN|-Sp-l%nD4*1p0Q9i8{t|m_#VK-bsV3i}Q;25GyVwCbJqQXaBe(iCpluwTZB@-$y5s4NtPaVVD3rB}xe`?PIf4k*I-0Z+y8&6&uq0_elY>}8`qWka|N;)~ob#J~YGXC)Ql0kbKS6@Ph7;?sdhiP^kZF1G;STzOIL=Na1N;XzdnOU1k0|9xWa3>Ijy(sJ=36l24yI0gOZ}E4LLaDxZuj)Lys&G&Ugk4gDpeYSfhm&;J{RP?vEyqcdm}p6!{IsMrI1P+HiK;To`40hBlaf+1fxeQU4||Sn`w%jxT@G}z=~h2N-g>pf*)x=$0Q4|?yEc8TMcN_t>q)XA2bQ_L=)q4+FQ)VVA{;8E$E5V^P{246-eCBR(c45aQ3M_;uv<iFi0E6+vx2jFga^Z})8h+8!K&bG;y^XP{gc{;+sbN+I?DV$)QqD@=dW$l9UZgOh)(J;>!G~o-KB%Puu|@%;m^hRcERDT^iZlx#GTZ%hQo4+0gzLT^S~+TocCEN^t30$?{`jM-dq^liZ>aJo-ppRV!uXD1BpXX-Asj12sE0Tg!6^1askPhq``p+8**Hp#vT`FWj9M})3)uo*Lh~>wu&c_p5WHbGl(%nrn9iUZN_Y1-dEkLfj(!vsPhcwW{D1p-d^@>{$|l1$lS1Hdi}Dh<#3^GF<B(-*IjLYx=4tUNOVdYNzjDZ2k2`yDaY7gFangH$w8KaX3{)JLODa_Cf5`8xWCcndL4zi*zKWfb0y}{g{w)lE>|5HdKO>GAT6crrXH({o8e7~mbD~~;?zVhQyINs`&wyf<UD~XnTko%qz`9vCmRlzK?`UZY2=jZD%I$i(rSX(9whJYswTrlbtfgAcA(6^@cqoHG!BnEFtP<in7jK%fg3jOu&5m>C**s+b6!w(&k$6Rp4|IgbnmMsS(8t--@}P(3&*pPf@>%hZaC%2dwS_8u(Ca63>eWA-Pp)YZQJsfByHp5?dl14PmBz&`d+hCd5e;J62f5u4?wqz;uCp*gjOESDjTlQzB^4liwb^6DX7agFfQRNx_7$OgNCZAchKjiNqbgBF<XuQkeIf<306B|KmN{JG{Yz2rjj|%N4O1t32%}9&w1?TVyB9qfuwR9bF5;0tD3KsBt@Soy+goN>-REBmLSt8bc%McyEvj#D}aM7)-eW(%ojRmdyWk!J7jWUdOwcQ@LQC7VIFG~ew3SO1)9>v$>-DN+*X!{h`_@+8lH!Y?r{(wht#*OUgjXI5Cb{_LlOzqnrl~1%afFF@>Qa!UBnC$8{%?wdh1+IBo^E}c@ovB$N`YWx7Gw$u7KE?BHt#hB*Z{YS{+ctqlx~v<#&`GEtTJzZ%tgu%C{`h>E(iP7*)!>C;gp+ZI6xsPezfFB)gML&+ZiK=sildQlgk*9ABV}2fJO{#cI+n8bzjHOa`$nb66=<7T+eS?Le`k_B1~<)D1>H!8J@25anw{n>f9e)S0cz$a|<Zj7a7Y(ZiI>r7w<{B~Vwkqsl)Rgb9v8%dV**=0h6ew`M}l;LD)_g5mAD@V#Z#%p9TfMW}+ZGPa=)3nk31mys~j{V_dg1I}T!gpp>y%cZnCy&TD!ZkaCLWcfPKL`!1c3#Hwv%vtgfI7+PH5-H_oRCgxrYnF*PLHA>yeZ{7)UT>+ZS*#Ss!hv<vZ)A<RBDp5MY>Qwq@wJ1k!*tFxxz6!r+7r`+D|5x+c3Za<%{JZxn8ON_DenuelOW$}6*Ek;RzjDP8mv$M>?9S=qS~YbYpA{%qI4>~OGyG_ZPYkvIjSU?<?vzyT>J>EQiheXCa3S@SfQ;xzwaX@iW)1S)|MLK!+=bJduzHkl06dzi7wKPy#Sc0frk3!$~u==)h5YQ%obkCAi;YVYO&W-X{&QH+j8~<RW9}zj03igsTK<2{t8XqeXQ?9DUCRqYHi50H<h)k+PKt>6=QFTq)N#GMBdbg2&TRIPfe>7JMrmuYLtHU)V?IexGSwygmk(sN!42;u+jYGQdGSgM?hi>-H@^N_LSL<X5U4-f{nP4mX|##QgR7=8!41L6Uq~jqYBfZjHk-rj!lRYY!tiDE276$WfZC`Yr*fYlC+Q`2QJ89M45xF3&s5GlxDXqYgG-^I|f6JUWh<N=c&MFfaEmM#~4VF$r$^#6t(VLZE8sfA&450vj^hsu-;8V*wU5qj^Wm7076f5%011Ap;0%<m}dWkO+;I0vqQPy9$xoVFJNRM$z;>b%(<j(h~Q;aXDI-W_s{-xRm!3PkNNQpif?ONp_$Yt!j%>a8?7OzwhY@2Y7dbg<rE+NhNms5B`GdiiV)!uw!gwVWMeHbX`jm(wTdr@H&o#{(Xq0;2{@vITl$EUsFpQy<}5VoKlAO1@X>0hA34{OQ35+EP4zC%6hl<<>x@5roQ-s5nd>Us<<Hl&ieTJB-XIh6Qwfu?#@nBplHRZE#cC<U4RptzwP3!QkM`2_z%f?odQWdUnpBYmDbJVZCz`J|sePXT)4S()8O14WIk}w%ihZ^UI^<VO>&wb_+dkzic7DMtE%h+bY*$%%#3^|UBJgR!1!<Jxvi(DRC}j$s#TFHAr&0)Zfu$k~@8E1xwhqNT+gJE?sys5hQlg>in1&yPNS6)>$n3H5t*_w3%~02UHG0oult=U|DX=%NlPr`-F|o&C&sI&0&dN86F0JI3$oWsU=;|n{9Y060ttAt4h@@$UBd}W1rkSo%Hx1kY83-L>&_hMoZ|Zyy8eLJYibUP2O9Xq*!|MuMCN!j~WrPO0k&7cCs$MkJMUa{@bxt*;=QH}e=B!^=Wxp>Z%$Z1CQSKp#Z%F5Tz3zN%zT)Amw}xXY^9+^(RH|4~QaL;mbsPHFip7iM)aXbgmrorPWxM;EF>fl7Xj~=wMXBsdhaLM&U3>KO1a4idh*$V0aXG9;$r>xT$CdRs1zkYOgcp4evri=_(I%g==L0Mi8|UvV0jF$0o>1gKHF*dUL`*+m+{G=^=d0qK1+JAcPjW=t8$#v$&<jmeja$NeM3+HhD7CI`bsUW@U6+`0tf`{~0QK^Z>UC?#V@H9P;mT3GY`TzOj%}xsAlJ2AiK$WQvY`53I%YC!BPOA`W88W&a&Jwf%`558zD5HC_Xib7sAH)wY*4ITt&C?sICP0kXr^zc_`JAP50RRY=|*Kz#i|-`(V$(aDt4&tNktyLh1kW<ISh2Lz-LVz%#W0ILlw@b?I<g5(6j6DLC?^-jj2yBOdFDe8(|0TSw|e-j)#5+)>Y|RH5|26@>yY13caGeYI=Me03uVZRUUTL;Lxo*X=%Jgn}-i=moDS1pT7u9&h@OcoglyfYZC2D7*foxOit0<i(?<b6HLM_(+s#39DH%QfX#$|)3(6{Xe%nxP`l2cv&JxN_43W@5{Q}%+;SCW5To*7WH?P_HBACqK5D68U3Vt-)n$B2VBVqV@%3D**RUXD;nA8Uv4LtHMG2`g)pkCiiw@zh_MXsuA}1S1TC0}QSJ4KWWw%HE&=o94Zj8oNoyBG>tyjn{XcKDH_qK$@Xm{TN&fWp)_yZ%dv>d~n=hZoKt*4x_6|)#-oRI}<F;6@hl#UHhqZc{%vY|1dR-6ti7zTbMguFPh=%w1-?n~ayG;?AabP*OgHJuMgZzRdg0T<J{7YC-Aa9Kh)<+(Ed>1b*S>fO_tlqOR!Z{Z0Bep=%eJqrSQsU&bm&|GiPuj<%CmLsT&c#`pM;gP*I&J%d4c!F<<c(%vP_D&RjFV7#_49-V4Kw{r;Nwg(kp-ixCI+h2}a4k8c-{DPwwRc?sTmVCNG<NhHJ*ZhX;8A;k!hwiFW^%BnltZbBB7ECgMsxKB%rq5JW%zqNAT+)_F0Dsyo;BkJE<OeF16qQvIF`~zfuqVMua)WZ_c7-qI(^Ml;T68Y3Xp`PBY2?+lzIoswTBfg;#1)v<H3O&d6REcO>ew0;qOv+Pm}0p!O+SyK%<vb5-(xmo0ZYYGWx71wN#z+MyHD8CoB<94b+l_lIu|kP-ux2X~pFU6!P}-=e*@a&IrGKnzw?)WA!w+EPKm;d-|x#5Zg_jecS~c^?Y66a`y}jYt}X2rGNXVfgMYuT712vjg+?b^uvr%@f@33_`w?39Q*dHhOV~MEfJ-?(;Rsdq{CO0>@j&3GI<{}f33*zN(E5M$~}cp(_%fTwx!>WR8b|16jo|b0(z~?l;jZLVU%ZKV9MA7=iwuH>@t4>vHXub+Sqdyx<@plbI+LaqCrSt{FD;tPRexQ!s}nlR+)UF`$)Ey7{qczz7_FDo4p(RmB-gv{%&UZ?$gJgdS)5^5kcCaPU9TJzxy;;T)iJNU7Rp=62l_e{M8PQ><)NM9Pn?*HnHUZbC%X4r^vP6+AjU+H{^fKexn|ZC{}ITD%*h$kH2sKe*5?T55DV?-v')).decode("utf-8"))
_S120_MODEL = json.loads(zlib.decompress(base64.b85decode('c-rlKYi}e+a_xWNXH6mFk(rT#?*~~Jdl%l_SxCAtY-2%CoR&l{ha8X`c^88I`#n+3M|Ea#)ZMJd_3RA1R(Cb4vNAK`#3SSJ%gOEePn&mVCtsfY_3XzVZ!XS1T;6@S*}QuH;rq*r^H=ZAZhpMDdUbRC;cj#D>fQC*&E>0~`Q(d}x0~~e+l%X~)7!hv`;#vz*%v22Y|i}N)2p+0o7<Bw|Lf%Tr|b8p|NPazz5eDua7+K~&%giA|NQFo+uyzU^TU;Y{^tMw>FeK}e*5P)-#uLVkFUP@)9JV0z4_+#!<FBE{rdFlZ(e`($DLdM{QA4oufO`2$G`gak8j?5_q*S}-ua_{`RbcLzW(Oh(?5Ro&A+|&A>Lz_o6~>)-RrNOzJ2}GH{ZN@`ug3QKYsPyo5!!;e)r8+|NhUfzv-v3z4XV|e|Yn!$8WFy^<STU`TZaM?P-)ZZ~pN3<zHTZ^WE?N@caLK{S7W*W&gUlJH5F2d2@Alee?In7ixZZe7EM8$JcCr*?HsU*N2yHe%Q{V{rl~d+E?DruDQN5(d+l`PtUF{-kn{Z{_C4J-@e|x^!1y6-@Wkd?_R(D^Zljs8|>f1x;$QZT#v^qkL&PwWoQ16S0Co>7d~8FUZ4MU^Y;GhPiI$eZ|^_9yT01|-9P`&RrBUgm+s&E>B8fi<I?V%KU{cv^M@->Z~ky)_st)!KE63F4Bs4IAKx6Gw{QNJFHU-fePYD^E9KLhv#Y;`bV}j;RBumnyga%8Q?iHO()lUzE&e1t{a7A;jQEl6f5a%?pWSXQFRnJH+u<&*-fsSO@@1%h0JwbY1#bW6;_m#X)4QK;HelGxw|GK_XWZSKU0hvU{djr@4BFh`A?KGD@856Uo&t*AU0h+sx9D{Gesgm_5NOdM>h$*g=KSQ#Uru}haO2&@)!$FwZr<Ph<j{3>y1Bi>j)7FMIKp3?TyD;8`k`*l{B1FO#Gie5cZwf=_7zko^1p2#aq=bZ!1$-<KV4kDJ-t19_x{rVCgG26Hb0(Tp1u42EdW*fS)5;AZvW!+`@i4aZ2G6$ulMiid-Kaly?A@A+4YATAj&!R{^ISK-;QOYeE;0x)P2k^PJZ6pI9lVv&w8?*+xg|$Ena}O-~8|Tg}lJ(p566-i=QsfzQ=Mty_>JX1I%B%#Wls4*Wds5=KSvB=l-QPKVlnu$#6*+3F~)r_T%Ohb9w&(oA|@o<?UufRD6AX`QhEwX}~S_AL;2MKYf&^kNWg6K7CA2AM?{kd-_-gLkx!-j7Nh3X)q!UhNQumG#Hcyqtakl8jQ<>ad|k;!MHpamj~nWU|b%I%Y$)wFfI?qmBF|&7*~c1JQ!C7<H}%M8H_7~ab+;B493;LxH=eD2jl8+X9wfzU|b!HtJx6<fOmZd&^tZ9zPjxZ<Ic(HFHaP9arMLXtuw?Uh2gWa@6TtKA2yFfeg6%t-GAyI^6P*WkGP4Cxk-+>Nk4iMjo%cH87UnzQa)xRt8S9ryF|usEA?$@=Qa(`Y;a8$iFWRDa38&|rVLWvU6TCFqENllad(38IZ~`Qn@o1slb%@`s&|}{?%pZS&9dJ1viq*(xjEK5UyAH5Pbtr?kG1zob#HlYrlq+z?RzhW7u?G)xL00quQvBq)C7%<<1;rmWD$Z(_1(L*JcCy?X;9ha(`3(qR!tnK?9P`s@4oV{U-ars59z(;bpP!^IjJ<ZUdcJT>wmia>E_}J7VZ|L`uiX*KAgLV@YlF~e|d2?c+~d$y$o^Ac)0W<M9j&TnL4bnmCTfKNe1J0puS?Gp!+@j<AJ$+_xJa%CjHa@>22sA&R}lioqxbv;&nEzSVDPnrFZ*xe}ZhoUqY;HZd~JiSk+i^7YEn>XLIr6Pk2anJN)tL`SpjZhl@VQR%LDGOYMB)?tEkHSR@xK7BaDwo%uR>*b)<_kr}H~dOhR*;_i$=7(Wa)xc~a_@SLd2_tzetE8*d<?w^SZAix{{b^NJZJvD~hnUnC&b17EO#f4wn2k&eUHyc!T_eWC~#H3hOPC4dXdKBy8V0Qw|vps4f+aokWkp0Eyw}+~QRG3nwT1Jk%yNtN-@$Hf7sR>Lh&Gw|U``DCz#lJ6hv4-}=nyn>yFx-&tZZF8wO3Hx@n5gPM582Q1koiK8$@lkFk7vZ-6rKuoTB3HU*rDpskrsv^WQxW<uN{<)G%x7*tSJ_zsR+Bmos9WmNB1DXLkRm70gQ9rhhKgJcJ`190V7E%#t9mN^~AoYT<{%&%57;P$xuNLLGYC_wPbi@5rhrw836JgGNOSYo*RZzwcFYyNeTRzQzQWBE<c93rOXzESaak!9}2ndJDfFfmsnhZhVGMK&+RHdfQOxV&ogbx5sN-DX^yF#bboPg7{c=<QB165`!CcKpDRSlhaytePAp-FMjNu4(k&bgOSQbz*qH?`mI(Y#B`z&5olr}Nx#W>8i^T1g^%wWXrB}cu3Mq!?5QM3Qd*XvI%7qAF3eS!(ih?ah$-=4<7A6K@l%?SMuznJ%68)C`4-KDU1~Ny}l0wmi_5ZD=gcYAjQ+h={2L}4aCspEw_CQ7O$tlCkO1KBYJq&|Gt+raAO6oWyf<V<++I^c}qWZQ4BX=~-p+3FJ+-b@di{!0v*+<biM6?iWglwTScHExuq)Wq7=(s3Bo;$nb+L5w=WLgfBK-|#HpRo-V!X#yl^I;M+KiEDAYRycTBwYfcs{$IFgKP;*(k??836!`bC~bKSNYY?D6C@?;`V~!j5A>5ml1P`83b)$so}d&Lw(B%?ss`<$L;^7NKFd$gu&c3{V9vruz_ha^sB5*EP|EJS=g}3xXEdrQPGQ;M$z9GV!K&(oaASVun%MJ!#18RWp!QLrEkD%|WfeqOdbxQg?8qLAnI@GOS26EQ{NGIGoig<KAW32hQIkEc;&k>MW0j&Bm$G$4@o`v{mc}aWG?Wrx<3%<<O<n=1z)wbVI>H+d6ftveR&vNeXqVt0$Q@iBpo)~Zc_s0GAy`H3QqI{Rg2=_Q)@W5KqQ`~9(26>NK9$(W1bEnt@Oa@tq0%FK<Xr7ufy$3v3x!<5Md#&34~zZrSk*G>B%cDUSi8Vnh+^D=NptwX)w<%sL1+a?m&h>B)ul`=#n`NRyx9m;<(Z+X4iW%}V^siFoF!QwY_sUTZ?Q`Jygtu}Rh7G(tz!x9ijqR6v9ceE#QCWm1NWyBa|8>#g67eI7NJlH^Ol`=o#hp+XFUj`K<ZQk+8W@z_nXvmxpv0m!r>rQ8lz9hH_frJY*}o1V_h`lnPLcDk;8CEF{-2}#F3^p)^qF%OlHWWpxGiF6<J^l`=g|E9x)O^^w*|9X&rz0w6<SM@3Eox#>`z(jbhWnv$>_%eT+#6|6?&}j@(jx<dzN#Y!axW1bk2UPm<=ekfc`Agpcq`)p8}n(iCisM8*FDNUb1PtmG2QOr9w}<6tN_Mj^tkZS~)mN*3-+lu`;Q_wKK~BufKPWoe6#I%v%hha$;?4^}bxh_E~9vu|oja}TDx2;c)?iwR0hQ#+?PbZP?lWIrG({UXlpoZ{FK3`-yoGpw%4L$Q!^Dnrv@p2^&k)G75RI|U;pvB`Z?9ykliFwRNYU6;rCrYI~7Ret2<C-4O}LWC|$t3Z`*mCwbXO1cOM52#W<HA;b3fhuVBA<hL=0;7!F*PK<&qXZ`RqeBmmg}}7#F7QK`1)z$o^C&57+<gGT8c>xe7M%^Mw1egV+eS)^nPZ$Ds<`ujY_`PEt5UWjp^BD<DgnW%pUxLpeX+Lqm%=L85K<P6<pcw(N3fQdqRJYfoN7tSJV*sBHjfwpDF#hZrjFx;M&;^MQK;gaS@I&`oWLojY@R&mc6m;#09Ax4&U3lYVY6GJfJMy_W^Uoh5By5T)<Mh8R1`{^%8@{|bkKRfGBCPL%+hMX33Dq||A0V6E{`R<O2t8^Mk*N45&%%pY_NjXxY1`qs@x?dAQicinDQ#dpNE~(TsTF5Vhf=p$ss7q<S+%ok0Mq?V&K@|;Jz?5d)#^BqyoC)Un**>%xk%pXOBFuJn8HiU|K*6?(}Gqf|i^)_*~-{M#nsYS1})C0%q}u9PBQvK-D%}MngsTxNw+P9r76(RgxHkkSDG*Hm24#rXJx|Zyk+hQapuEBX1RtWv%z#&uA3JVyUQ%K-PU2$i9#jx?(_N#-8QO5H|IBg;ncnhq=`uW?)9UUS-L+6*Qp=PQZmRz|v&_OJau&55*uU*JWEu!3xa>SW+pJtR40VBoEkX!*RfQvLmEtip$8kFw;C>#nxqsltm%kvAQHL4y5814$VUaOhRF3JWZ5B%~ZH>I9SyYHAJ14IKZw%Ez(BR<OQ;gW3W;2IssHM^uy%JzF?A;g6sn&2dTxtLV=WI2&aTms5t{1!NL)&ipydZV_C9z%qKKXwZ1ydN?^s{D=nYFYC^Ee9T!v5oMGqZoOyb%3LW9mFe{2+Dp_U$t6H-%kN(SCQUtqsM9><F)o4u3G+LDvnEa6q9LGd9u~x;9FnG3~Ktn~LEJvzUbH}Gw5{gqts_v=;+ZxzY)`O~8N|;fv;?Nv^8hhro1YzbB)AG_eFn}u9Og5l!m%z@sxAXh0-Gz}<yC%|MbwNCG3;%<Zvywf-q&==mwRh!+jK6y0xWiD%(`m90ppR{ZK;>CYKr-$iL*0Y#l`f*{#3gs6Krz*<iG2)fvY?3Tm^&V=E0q^^u0!}5`I&QOgK(@<sTOjsPm5Hao$J)wDN+c!KB@F4tkZhe1C+Lu5^-^?w>VE(muBBemxG6ZGM0~09jpu@?q<(T4jw?6W-9BVxg%%|5@FwFcUd)Nb)BGu3y0~#UN52a@nMkEB{t45(ps<5QN7BRMyk~HCLri7s5EIM3jvjF57j>d)igi_6NuC`e2$*+(!}|_9H*#rgfix6A!XuaaLPJA)(pYjZ^g4sR=_FXq)MC#r?i8e&OroF91Clj9H+$B(5n%!lCLBmiBn}^S5@pR<p6~$QAtgq+y%`)@sODr&jFkK4C*i*6>za9K~e3SO}ap~qp4C%Gns$UGxLDPFh0OrsmXiM+9jdC<({pN0D*_<Y?Uwz3XM~pQcd&8Uhbz#9ZaI&R$hWc5zA}%sfaQ9Jk}qPpLBU88!0KlyiF5(s-kp3A;`>g%CS~Qo&a)K+*M`TpZq4z09+5Lrq!xOkg7fUOi7+AmQB$XAo##J7C$SV1F5_)6?BgReYtoBHtpp{I!HVN?G2j-6QTz!X;N!OonbFd0;Eohdb9~+?=m=AR**83xi7&zkE_ay>=J-I1<9aLX|kf~2rHt)ieS7-sSBQ=0-o>P_ca_=8)u`XhiQ3F6=S<S%@ayz7S&Y@!0bs~pAEpN48YXEILlsy2r^N!PHg~|zHUeYNO=XlKFVboKNUko3?V@SOpPkMjQIyWJkAsasvtX9wUT1)^@9|rH~(^H!Q70}khwHzHG7(W%*(r}7~tyKbo88~V5`7X$>Fn`YC@Fqs9F_+S;nQwQ+_c>Vficu*bbTvcwtL9sTf=1b;^@u4tWib^1}LM9+2|NCfVKSSt(@-X$Buv><Ssi$PwEn+>`>^7o`>sRtHF>;7+RON!Ao%T+jBK5~=oZQbD=)J`SKfug_YSRanYP#U4*JmzQc=eU_3W3vS;|?xpgR7|8tM?xIMIJ7pd(m39h?HevCSd-G`(Cl!ElCMOkSBzyFD3>p})NE7?0JU=BU3@BAUTx`<I_w`Zf!ak}DV$Orgm;X4?RNQ$5pi<Pl7&qk=f~my)>2u)}j3ferl*nG&{3*zDx@anju9g-SR;C=1%@R%Z59p#&w_<XFzN2|Gz2N08Di2f#yF*2Z<O<eNKB~FH)SF;!fj;QLpU<>ITPRw4?kEL{O$r%<(lH^J>rEvIDsho=+}UKbc;p23Uv!5l^o|t0IC0F)aW@sJL)7Uof{)aZVCyK91f95Xggv@g9s(5O5y{HUfh~zgQ$<(isP9#^l5jNx70#8i9JR})UP(0qP-MSUrNe5#q%J1FVW5zqnzq$}xK^%sRJmHZmOAhX3gIW^tlGoxk-hY1@f0%Qrj_wh&MLApQ(ZmJ{R16Z@*P!5FRiT5^scl{Tp2C|kb{aiMM%|e$EcI`WdXc1c?wXh>L!fK=);~&IafzN*H$w{ljtS&BsH~lxb|0ZI?M=!QsvgAg5%A+err8I?XBW*Z`IOu)H!t$lc#Ve@)&zsqmKHtiGL$Xl{C`X5_u4?dUT9bx@26U4-=$9NW}BhQTHarno`b`SW;iHWLGRxMUA*{piWgfOn}7Q2kuvi6VxgG?dTFuR`tM(Laxk7U1ZujwJ%T&kz;?3FiSi33@ZnZ*C3!vE6tgz)Y^{Nfg%Q)DU7I`A)*ONH|-^2TA;FpomH`8&~xyHvqBs{-GHjL<%40Pd~St<71I?;bMt)hRIf4={A^Hm3at9v=1hdTGD|qjg7AV3+4E?<I!P$p1j1{stIK4b!d2xpOR}rrUc0AcTf<c~YvP~B`e}e|%TXPVZE{zYhYLbH0|%|Zgty~dRZEAd6Vz!}sS*fIRV1xf{>5Taoe!*Po0mJxNP$Tz)*O2;Iq{TYw7C#VOUGV8EsA&3QmUy^Yay)%3TY^yR02$BD_#z(^?8hG0WUrU<gSR9Eea7<U<&aQ3snbOdcpHtyyG8c$P=FevU;0v$-w+XZqgig3MflQs&mI5$*201W7WNzKykkZtDlke-CUM6C;C})<}5SSe+gb$Pa~y1fg(xmuqu_qsvPLBN^S)Lr?7KgM7Rm4)>=ZVO%SQh9az|lt4Bx~uicZ@(5+%on@zWJzf8%&n3zeaiCLIHw}SQtg92jGvP8{Cg{liJeGC{n&I%O+#6}>yoYy%9oVlt(eGFJ2o27Ugff2XILWh8mS41-%0%~Wl%_0nZffUNqp8-ngL{sw2e4j@y^Pd4y@{Ut}2AFam4!2%_jZtKp%efvglWaoK4K%MKpC&&9Bv{iC7N#HlCu?i>J#z;5moHB4H|O-?-TUjCyR)lvN2y;<zP`Tv@b2oA`i@)oU!<?Ud;cZ&FLD1R^^?B;QhJnocmVaJ_V7)He@MD5<?!%z=aH1Q3i!j<JbV+f;d3|)M`YglO?H0U8A8(VH_g5*SoZg4S8va59ilfsUf*1tpIx4OdG+D)vd8wD>koIE!AAGrL0tWR$xd(1uKpU*X*xe8zCEQ7E<tj)*B@@qH+bUJ+l#kne*fF;%->zV-CX)|U!A@4r#SlK?|<IhIDzBB&w6q%FfPw-Z~GDbTQ4QvZ*ER6K$vHD{ogv(oqfN#93aTMv%j4_?(f^bZI2_bzyI&e`Q63O{abH-1iSUjbV(QqGU^7x=oFKA{{dw3!`bES=7~o9E%6fn^{`~&<Y`BqKJv+cVxE@wWI#Ft!g>6dpFY~jfS8_!7!EfWj|KzMU_=@WNrN$IFenX1rNOW?7?%g*@^GGmad|K<560!exI7q_2jlW!Tpo-ogK=drt_&A=Fs=;7mBF|&7*_`4%3xd>jH`ojbug|D#?|4@4#w5NxH=dY%=H&1_p9qWpfFJA3OWviz@3xR5QVOOxE|t9@Xzlqt{&y=mZ%@Jcd&|o>L2p!fEJIqiI2HSj=4!cdJ~P`6ptAx9WzosW+ba_lHI#J3~}tXQs0(#ZqxA02G?Zar?P|l{H%KPzM3+~15AfYlAl==s&_i>PB1=4iuGobTSUY4q-U0f>K*TcDTa5-bF-|sz3jegd2WvN&bz9zduMreeXPA#s(Z_GGcC=%Y2SM}yx?Ab!M*Z=d$qZ@q9$^E_nDg;vIu=e;@!JFR$vsbYSQpXmLZ=edk(Z};z(t8zQlR=m4E%>!4Ur88W(oW;>w37&sxQFgh|{6T_xByuni@6ycsWD5b2d=D_va9Z>Om%tP-e?SP4ZJj{?}vcr>We#+-I`(D{O;kra};#$`A(v<#HdHOa(#`sY4o{$=%4y(WsFc260P6p-4NO^$RPwVV>WSlr?3-MS>!loLYg76{4=*k2=FuqaD`g<iZF<+dgZtr9jPLOkY!NMVF!QaGq6OJYDmbK*jE=l~s<go=IYN+Nzd`KtZrTGvm<g~RxVBvY505-)vg=-ouW5eDL!8K`a%0ZIYlmZE;L9M@?TunGrh?gZLHF$>tQHAGGycm@EQ?WBl1uZi<W>2czdB7*n<>Z&{p!IO(3B7tTQVhaRgT@Ko_jHRVff;380?)w9W<)zUoti;w42J{NN1rj)OUWJ#SHS1i`h&{j5&(-cpJxwo|Qt1q;{e5%csbvzYhlP8X=y7sM7iT5vz)7A(P@>03uEI(L0J1b=8uLhz!DKn~>98ppuhCinq&wOdZ~{Jp%z(ZT%zHLKZup_^)QtgIyj+`?a(6~*g%a1%r?_yaP%`bpHZ6pY<EFSYH%04`c_ELeUhFEQRk&%=uE2Zc7HIIoVnso4TKiF!l1m|tpR!LPm$dV)Ki=x9Z}zOy%cV%3uIW{)`^NWb(+j758UlW@eiChx++IwT7H6v1WlxPvkWeAig;5vcshQ(}n<QAzXB;{nc!eZb3ZVI!RJ~U*#7?{|vB54>YntCM*FBs0zy@pTb^ITnFUK}m`xr+7b2{&>Q<O2#Sk<9@I4UpAQ83C)+>r2A=d_ajG&f6yMoK}w6*t>NgUEW(x==6ksNFb;=BMO6@T{-&!Lezwqp>1-TsVlWv~&eVD0$(gFw#>h))XHP6;qIZ9dHNO7gH~9k76oYnyDmpeTAK8!oines<}IXaur@F5JVsbZ$M<B1ZyU(wnXVVZ=ohL>3%M0fs<T3IRQ5E<dr&JX@)Tce$wQPq)OvD7!`#F@s!7Pt9y;`e_<Q1e*{aF&n&fo>i2P$TB!WVoE-HEd+IK8KZ<rQ3^zqkU#$C4r_gOUCeTdD+7xPjx$0Lab!1*{Bzb+EbR0)fSe&KAYmIv80YJ0yD#jFhxq4o%xDJddh4tuJFBYh+uNw0L)pcfODfR;#FE~T7udmvaQQ_Jj#|KKO&_O(9FU#cvh;eIAam`c*iYb5rfT5%$NdonGBF8Y*;?WTTotwNYpuo6bTS-o7jvN<tQD`jY-f}NPo>ytoY6%i#Hl_-G&ZwFpsW?Za>BW?oH<$+y_kKT1lglaV&pJYaL8urg<l|V1mu4xJ4q6}RCW!YWN~?I%8Kal<nWd&<DY4kGC0EWYq_z&+6f8w1u%QHqEV-?(^m3NUm_zha@C2y~zCF(AM~g>Ikag78$WsV$lk_P%{c8Wj$2n>-|HO<eMIA?o{_K6PIftk6(bM@S#Cvlw71&giqQ#gecTi;YoKI&e3R*;uJDQ@!!zZ92xX0M5k^>UrN{3S{Iq_#PHKCZ&++_}?Q3`CHB-4v2vSLx7{dq09MlG8_N(Ji*7np)xQ0drsBw9LzqF9%U!TL*h^_*p$qp2*M@`7RJ+F`)t^A=E2s>*UK6hRUE9CNf<-6VnLr7K=Qwm>c8z-fjM3g8l(f}<>vt7el&P^2uKnvi#>(j4SS|HRx86m<;@>vbGhGMTN9*IJ_r#@ta!b*GZl;Tckly*D-!tyfVEM#wl<NgUy-ZTYLck9&cKV&0O%pgiR^Q0}Qb&0~$mc$lP8?fhg)LHIV-^1kkfLg~Qhb6`T}Cmq7xSSB?UMo6SC+9o4hwR9APH^?g?VM#2O5zFF}5LC71hn+I)6`&-jAJnHcR6nRRk#sKA4`#B5;Hd`jbDhc0ro6pDDqgUscpYmA)5j&HkUL1hXu-VXlOE}zn7dMOBd0=VF)N@fK@*8mVK%JL1zMn!`|!(xx*vJF#^iy18lB$O$?8z1(aPL5Qs(IXpl5H*FLVrzFH<na#y0Uhbq9e%8(wlyEMZ9A>{7|=>Frar{qS0Pu_=NWV#&{Sfm=kY0#Bq-{Fkw8YWa9Jbz4#03g!kN$$jOo8VF-0mT}>r(qb{^&YO`%g58rxI?RsC{RXDW!Y~CRAQmsSMk+CCDyzU$3RY;rAb*%|ed{xeFi-OeB|o>R(40rB2Tp-gk-bYc#~k@SM|V<~qM}^_u?w`9T*Qo*fhq2CpPqqHqe{`Z3QkcOjrf=Xr()-kqD&e8A~)v4%W+Dri_r@avf%&P#)A9d6o-Y)e)rmY<$mA#k+hs;%>JMyh6SS%1mJA+Y;N#E7D|GEh-D_5pN%_j6b;FllZU0o$-UFmc?oqg82$_srD-m&RFOO`9A=_KZ}i8e5n2GutbK)l%~Nw%(XGW`f{b8~BFmh)^d80K3txLB1-3k25#7es(IadLqR~(2Dx=?XVCMF8u$P7(n=$i<;J;^peFZ-tBg5qPa8B=Y)*R^P%EOhzLGnW#X)}9@wV8%C?|jdkiQhW<)}~i8^)(|LBwD%ezRx9&zM(XZZd}x{(i?h^<kyZ0MS*k!j>OUilhm!0`Ow-MIeNwPl6(y4`)$l4*ZrV*w37<YjZ@m-R3Xx5!X7Ly0#lZf$c!*WZQbZVu*t#3*f5UO*<rLL8gZl?C+l%m0hJ`W@0IB3<$brw->%kjkfz%~ix$%)4Ed~~Z5dk^Q{!|*weQp%)i}JbtFpO=BgsTf1?x(0&;`4kBmsmowlB6JW3fI&iHEh74}huY%~cb&jYX)*QVKOtg;AiAnSrWSEn@0mv#EFK4jS7oZOg>wPaHz?+bx4AcNsP%pbGl$efsmV2;CFCw}Ke$_{%z`Rp}vSN`R4qZ3QV7_jhP5I_qo!JFpFhm)e2BA4Bm@QZf=VrZpVu1f4Y9U*A{E$>g#o<~1w;@OmFRJEB$Tm{z3+pq08GEJ4kO0zp{YM#|J(|8U{ZOcKcEpcch*kW!|mo`%$8h<6rj(Mp#_tJskXU-D`vxsKCW3asWf|Nb$-3Kj&+GY{?Y>Hv9aw1Tmsk}_~mq3EQX7eOoTyA9`3EXBM^WLXba&{ao|7pH(LOfg}O#Rv+}NO1Bbc%>wo`fLydWeFdNS8-{)@&@(A19ic)nb!iWNqxYd@oI}#%pNqAG8R{)sZaf4yz+A+tR1ABFm#QDv@5TB6#E*N1~i6lu!@7#g7$@lXThsr9l_QCh(H5MZGh1vc*TXY<zmoLyx653iC1ZEw^ii|h+gV8ML#dh-erB2!y7Nqga6HkS4}xe0!8$lL=6UT;b?2rqs%2lou^CPD<<k-1QJYrNP^geW*_eVkUFRia@jlKK@aeHlzX)SI2f&J@hV;%#zTm=WwhELP^FoBD$d=bu5=939p(meNS2!f<tJ6>v19Xr(U1_vIrS0smPjld)lHN0%s^G=6?(NsFQJH~kKWUwt@OD_V&U9{a~pzZ*K1s(^y0N`EyhLYJr^(=O+@b%Qdws(tvwhrloXo;kC&5sf`a44Gh*;OW#T=k{l)FcKQ3ODQirFMFZDrS#}NtxbSZTwJH5CEOrs5lp0GjsV7S26_9!0{U?VtX3*!_}h$ZNAF+dB!)WjOq9ykRRm{Ri7BtIu>O64yER1gfK=sXEfL2^m}22+7K(GcXzk;)4_#^^;9EU=U&2X+pqqSB>MsFdiTc1=3<Qb+~!DbIye9=01x&;*1isjV3MVyq%q3&qvaC^1V;<w&S13)_AlhpJD-_WSfHzV46H!GOrY%_CD$nS<huI`yJt9up7SueJV0N&dyw-=`;J>KX!FNx4WdsH!mG<Mh@a3<`CW%HRR!rhWZh#-D_xk;-e8xD})^Wy@7=@h6K^u#Gf(pZfd$k*gq;Cy<r7kjh=ADFuOqCT=sGUk+56JCJ;IUlk0%K6Q5)paQ^q7Z_|?n+v}XP63?rTrd?=iC%)=>wSehJxo#TOcbJp#WU{sD43#!^VA%sB3mis*`;K)cXIwjwMBeXlA6=MpwKu%Vve4hzk7%P<Gt*Vdz`7|AfSZj$y9qqNxBRx0x*?BHfo%vWK?&NrNhpU+$#(F!2ttSW}6Mif$IKFbcFpE>RvBhRx4inT3~uo6%B}KhuMoNU^SEIJh^I~1E}H|FTpc}Qj-J7b?!g^R<X8P#=wk%KA6R06}%BddlFI~FNXf;?=N4R+;7Y2#k=>{H+N@O=MGc9JaqaF{RTcX4DYA*(5OA^e4}k~!$GHe+v#4PcRJD&?KDs4op$K7)8AZo`jyj85Ar^;|NYt3+p}AT)Xk6AHy7t;mnUCdeYm{rvH0ft!`-m|YyTaT(Epdp>CM^IUqd>L=a|7Q<`gbLAGg;ZZq7G&;?>)Ww`YF;+xzC|*KaqMzB*TD@BAr_*7*CMH#bg=xbU-{+yngO+3oFP%jO={-*0YCFF;3Ucm3Zwk(_<Mxg4;&H&B1vj<<i?%Jl2+|9f+Ocky%o)|(%}7Cq%(5=Medxq<LF#bn-pz^?vqc6qyb+ERZ@Z@>PunNOSW^pQ^n8~n7qPaosc$Mp0uKYg^PkL6@YAVcza`aBqr1|!m7NE(bugF$I9Dh-CE!MHpamxuEljLU;@c`z;y#^u4dJQ$Y;<MLo!8H_7~ab>u`gK=drt_;SN!MHLQR|ezCU|b!HtAlZMFs=@Fb}+6E#?`^NV6MM7xnEu10Wtv^SI|iS)^|=$L(sVT;kts0UjVvSkI1?u>IYpLtm2>ghx|IA#UpOwV{VdTZqkq5MB_KbV@68HjFgWV$*P-V_byMv9J{U5x22uiG(5AxHCZIuxz8g$M(?XBqYpnCE=hi7QK;VOxI4l494XeDO<wX~xSsUP(ontQeH_d1PI+#Y^|qJYcP-D&vEF$Y>ATBQ%CqZZ?Y&anTb`R~Y3@z?-pk<y_wozwl^5Kr&Ak;hk(ZNu=H`YhpyuZKZV$H~#jBb$dL?AQr@byY3R*RBq_R6-m@2!k{OcDFhU*X4xUg%+?(J-&V~#JS_Bp{I2wr0+3;U`3;*n7ljpnRZT^T^;d7FKN0~PXI;s0OedrTT!YqkcKu1TRym0JxL4(*)^w&`M1;<mP>M$4%ER;->j7HUy%OyJFhv`GP3BCYzlEjlK{l$!Q|=7GO$ih8N5vz7zxQi>_(u6=Obmj)NciagC+JEg`Q*y*Kc@pzBidd?QC9vZT8%yV|0)usram*se@YDTNL`Q1f68se45ss`kPDiN!7O2l!%qLuTk_Ve4JfK;qA88}y){I*hO<Cbt}Gv}<Gq>;|(V@crc>Hfm}Wa>3GyaShdhPT*y3^jKwQq52~bkUE9L(1+5uQM8!VknNIC;XEzQ~+3dc<?>Nr|B8Wz)-Mh9$Cs}U?_<8q(UR5^7mpXS(v4=*JWndDljj%MP*jvsM*_3)U{2t0|5ok!lsd#FU3%yjEdzc7)r#IO!AJWag5q0V5nR`Jr*B%R;FjD5KCv?G}n%mQo1lpCF=W<7LR!JOx^XQ)ZB-r4Nry6i<)8TS^MMSRd`B4l5=T&4C6z+2eNJjRh)sRLI;b`&9k0uX$F=G-pwx;FZF+{oLai03q{CP_9UqSyH%qsT_}GwA(dnCQUXC}B^f0YbD621i+UAgPk`EX=58C#hWaEs99l3JaO`2RG4#PJp{aAJ^-;D4K9r%DI_OX_XK#i)BB%DMl%E!7DTX<P_qL$PmTx`rYLP<u6-Pmv2ubvzePJEf^C|F@_q)g5CM{5&u(LKjPeti0YW6zX*{BJ|emP4ONI{mqKU`niryRvmdGQv`)_KXJVqE&ZCcFwy&E0UeK3|Es^A<@VV6U<R`5KAm0`ZY&>gM7FBdD(}7_uKAmAntZA5<&ZOYAr%QS!nCr3o7zW!l473PLI~n^)j1{L~)iFqX2#TR3Z%DFKKThQ(@PikWj>Q1Oe}l}vr*JBTWGP^==R*wp6gILp#`20%{Hn~qte>BSV_*rJ#x$eB4jO)jTEkDYhh@fhLEIUUDTW$`A?xl3NxH(Qbxu|E~2YUGr3V&2Mtz1dxpakGq^f{kOd$*H~#6mjsn3X=9D_`_=YCs<REAl@V+S)l3VlwlH3F3pOUoNJZKJ=>0j#d!+W6btAkvHF4UI=Uahn&U+q`b4h|<;^iiO4rO^EPKmSA?r+H3dAUd@;2v*(@1%EsR>>V?l>De*n=<}yrM_pLVMNOI;f2^pbd0Ms^;z2^KmRi3zwhwzUY?RK>-Qw1*uoD)iG`)oiRmHn%E#e8Sp~po`A%(%9)(Am8ry;iqL}|v!)O;8wEcnd9fe8iWJHcIx8-y+;%9`RCB4b`EcRog$H3L4qViY!*-$eW?A}na&r`+S5T)~4BihpNNOs1jReabhJ{qRpv8OPdaYF9vQM1f2v+gpSS6ii2*O-g-Jx36!Kz7JKfm462kKz%=kw4Wrt<ydFn|BzIdLi}$OFub2|~@Fw%&40j+73=sg$fs5YHC9w@;=tOIM!;r()%&TnDI<H!z$Bs6dNdrjT<EDcZ=f&&!cYx==FUN4%av6aTj_P|4B_#|_G+!bFs)e)5yoG5sgiseap^=8@eLQlOv$PNuwyJB@PEnQVW94Z%J&^{U5Nb$V}F$Q~-@MAH{o%G00vB!gO*{dCN0{IrD=_q0=VVcB1tdPIUm&gLie>xh0RAAx49<Wg5i&`zMZdt5SS*0gl}nyseWC-=`G`N_Rd9xdvXFCuZCypT(zzUKfji+3D|%{eXA4LIqu=-;cMA1r7x){tYcwo&4XC4b<~a~eQ=Y-S-JYoo-aaf-U=xrWaq9epA=)rPrr5tgd=9`&<TDJxhgbGGvfou>tGg3!-Ut9WV)CFa<%m+r&>F}GrWCJRM7_0l5->?M}45=uo0X(p6n^PYA1Pxjb_#;^QRlq#t!FfP60D0|DGW9^frqtc)_h?n4h=0ssB`(#p(+QUA<pvkL17W3kCx=y4grSY7m3G9;ynF;gDtH+u9C!5IWmm4UBJ0kO=H#H-*=>t6@Bxo`BDM+wwZs9TZ$<kqIW_<)Kj9-pY!dg*j3!D669gr1V3_UyNC}ka%lK#<iu#^_BdP<Oy(BezBJUK^Ujl6`VB}yc@Y2GveLnQ{<n;7LHZLY(f{AA1X6ML~)QUK<9QJxEjqzPsfy=h18>mjbt4t&c{9@{b#LqTOSbw>{DBa^_>Gn8~B36_XO3=^BKs1tCMXa`%MB!-ZZcqB)ag)P%uyqg&oUd9>>e!2(!#amXF40wf|!?M;ve@x189`jVW;}T0!%ur#S+SCXATC|J!2Utj6VO(c9=&vj6Nc0vzzA^VwAgvKg=+-IC)NMX{1yC=zkN+jFi$8V29Zr7K@60Ya#CGGx>h81^D3%WM%b@2R$<Kqa``Kt6*{?ZrT&|!L8&pjow8s}f<>bnTR#IAk+C$_tlPdix$1Vs_kmzTGO+nq!oY;dTzvNDmryq(zz)|C-Q>w@v7Y>vBY79)u+4X5pz~<wk{CMdozf4`k(ie}dWHGjXh|@FnK_!=>V!bb(S2#XXfC$Ya_g`u7xRxv+c+Z<^RNWUJpD6{eQ4O8VK<*Z!E^>Yb$*+A|=)xgIl~n?s)yhnVXXNHX3<)+6$fJm~k(qUK2@3}ng(ry65@BH(dtHxu+j2DkE{sWjURd%&T@g>nG6=v7nALS8f61f%q`Bhzp#`!LY?Ndnr6vz8l|_vH;F;PpiGE0CY^v*Y*=rz$ypLo)h~QK9J_|8>r#4^7v{%KAgJi!JUnOriE}BNl_3tP8Wns~eg;DP*N?AmT=}PhaQ?k?^aWmEt=&cmOq(j~Pb{WAh$wpm=lL&r5hFkicDeV!hd1}EAJ=z67l*R({H|a8QGa?;B<QH?>o~5SM|NVlWI~YG%F+wV7q&Mx0<$lsmxQ~y=G{E<Aq~O=31;5u(OJG%~MX`{rBKHC7QI@s%q2#*74?Www21jM`XIsML2d~YQESsn!KyZLAs$M8%udKVna%H>hp$xF@81`*xDq4=F4jv}=nY}D%Ns=%F&UReoA)Y-FjT)AU%$`z_M7dR_mqF7Y(fZ7=*|h1@FctU3EHVqHlva@Z0Y>>)M3dyPhGO}khGIq?1Tp~{{2#@=s&pUJLC-T~jRh8%glKJ%lt*3QfuYonDArz`(_m1G9%EPfFuNGz8oqvrW*PO>l(Bwd<&b`2ufT30W-kR1J=LiNsxirYh{^^i)R<iG=K|G2wIz*ODde$eG)^-ERlWC#J0%a4h5;8WTUlZ2g9cMd3mW~YOR9Yn0%*L@ICYia%NS|2l)(f|BpAXUny#gwEQO}dVvVt*s2L_d&>ogv6Wlo-e0eBPHRaJEKomd~S;tJ2<<65i5b->{CCCdIU}e33s^zSar#?^?PvSQ7M(W-OHWu0IM7hPPJTq3+c1mA{jXfMJme}@OSjbR;(Vxqb6-c<RQ&qgdcwg0L6$j;vVScZ?L%euD>0q!zQyZ7`at~#M@egB<Tsi|7nz})IdMJO(6IN$tbX0`NyXR()4Q`K>wVS)6G~OVL7E7Qsq7dJkQ~P3-u*cv_6sSBl7UEE?s_|hbV0b=dcT13|wf#Z%YADUNu+ukc2aND?xn3-(tdlGW4K!+8GzrPXEx%_}s`<frYy6SM`O+ngEV|>@`-Hr_nqMuJK^aB7NJ25roZ7lzH3|~x%iX>({<3-9ad(aSQGH%V7oJWPu0RU9Vz2H12wiE2aq4tQJU{n(a4>OC+n&mB3smJZP<;lfDS*lihZHROfwpFu9H|lvF%6=U%{`1|8K9C5a{^F#O0ox-$)^TX9zs{d03%k&#HPln48nl=Yne+{De*X%lCU&P#Xcg{tK|8??mC^pf<{<<bH>l*ruszmt{8!nM>~eNg6S|LO2Ot+ib>cDLoxuOamjLh$jyft4-jb3r;>W*4)#F3S}d1}+)H|S*gV7J6Hg)np+ZxPy2hd6oKf?to!3_PvLV7T<rh2Bn?rQ(P>cOywr%vtUgz{9Fh!qX>a#jEK_SnMJ$9O?^euK;oO-b~<vlfke?dH4v~ABklQ#8Hpjs;0XHKL_9X2a^pO{p9sx*1Lk8+owXctTD`J$H6o&i69`Qqe$V@@yLy}!P>JG(k}jQZuVjeXzR<^BuN7Hl5A>NYm`;cM!#?crPQU-SMOhrW;5!#COaru~xLf79$PPfY*)+11;#TZhTbkJmRB=VzBEUtWE<yu?fY`u_ke%W)O')).decode("utf-8"))
_S120_DECISION_STEP = int(_S120_MODEL["decision_step"])
_S120_BASELINE_INDEX = int(_S120_MODEL["baseline_route_index"])
_S120_THRESHOLD = float(_S120_MODEL["switch_threshold"])
_S120_ROUTE_NAMES = tuple(_S120_MODEL["route_names"])
_S120_STATE = {
    0: {"last_step": -1, "route": None, "predictions": None},
    1: {"last_step": -1, "route": None, "predictions": None},
}


def _s120_tree_value(node, values):
    while "leaf_value" not in node:
        feature = int(node["split_feature"])
        threshold = float(node["threshold"])
        node = node["left_child"] if values[feature] <= threshold else node["right_child"]
    return float(node["leaf_value"])


def _s120_model_value(model, values):
    return sum(
        _s120_tree_value(tree["tree_structure"], values)
        for tree in model["tree_info"]
    )


def _s120_choose_route(obs):
    values = _cgr_features(obs)
    predictions = [
        _s120_model_value(row["model"], values)
        for row in _S120_MODEL["models"]
    ]
    best = max(range(len(predictions)), key=predictions.__getitem__)
    if predictions[best] - predictions[_S120_BASELINE_INDEX] <= _S120_THRESHOLD:
        best = _S120_BASELINE_INDEX
    return _S120_ROUTE_NAMES[best], predictions


def _s120_route(obs, step):
    seat = _seat(obs)
    state = _S120_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "route": None, "predictions": None}
        _S120_STATE[seat] = state
    state["last_step"] = step
    if state.get("route") is None and step >= _S120_DECISION_STEP:
        route, predictions = _s120_choose_route(obs)
        state["route"] = route
        state["predictions"] = predictions
    return state.get("route") or _S120_ROUTE_NAMES[_S120_BASELINE_INDEX]


def agent(obs, configuration=None):
    global _ACTIONS
    try:
        step = min(max(0, int(_get(obs, "step", 0) or 0)), 718)
        route = _s120_route(obs, step)
        actions = _S120_STREAMS[route]
        _ACTIONS = actions
        action = _weed_repair_action(obs, _copy_action(actions[step]), step)
        # Keep the exact market-aware execution chain used by every frozen
        # route source.  Omitting these two transforms makes the router's
        # "baseline route" materially different from the validated fixed
        # agent even when no route switch occurs.
        action = _repay_shift(obs, action, step)
        action = _rank_sell_slots(obs, action, configuration)
        action = _preempt_shift(obs, action, step)
        action = _terminal_liquidation(obs, action, step)
        return _align_hands(action, obs)
    except Exception:
        farm = _farm(obs, _seat(obs))
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
            "market": [],
        }


def kaggriculture_step120_public_margin_router(obs, configuration=None):
    return agent(obs, configuration)


__version__ = 'eba21-route17-step168-public-margin-router-v2'
# --- Public shop-state Route18 override (generated) ---
_Y18_BASE_CHOOSE_ROUTE = _s120_choose_route
_Y18_ROUTE = 'route18_rank06_6c_12s_100l'
_Y18_REQUIRED_SHOP = "YARN_STORE"
_Y18_EXCLUDED_SHOP = "ICE_CREAM_SHOP"


def _s120_choose_route(obs):
    route, predictions = _Y18_BASE_CHOOSE_ROUTE(obs)
    town = _get(obs, "town", {}) or {}
    shops = {
        str(value)
        for value in list(_get(town, "unlocked_shops", []) or [])
    }
    if _Y18_REQUIRED_SHOP in shops and _Y18_EXCLUDED_SHOP not in shops:
        route = _Y18_ROUTE
    return route, predictions


def kaggriculture_step168_yarn_route18_override(obs, configuration=None):
    return agent(obs, configuration)


__version__ = 'eba26-step168-yarn-route18-override-v2'
