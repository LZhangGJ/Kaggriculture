"""BL-MDgogo-10C4S-R0: public-replay consensus route with generic execution guards.

This is a behavioral reconstruction from twelve public traces, not either
team's hidden source policy. Clone preemption is disabled in this experiment.
"""
import base64
import copy
import json
import math
import zlib


_ACTIONS_10C4S_3Q = json.loads(zlib.decompress(base64.b85decode('c-rk<U2j`ia{MoP)`Lk=UwPBm+}K#n$dK(2n}IMGAR7b-HV>1$1^eIQSR#3OPjz)wpF`SS_{juC-+R7KcXf63um5}Y@4x@?x4-^z_D{c_{q*VG{hQz3-+lP_>2ZDbbbj_9zyH_2{rBg;eE#^i-~af}zy8nX&%d6%efQ<B+J~P${pGj2U*7$EcYk(%_WEIScD`)B{_u9aen0uchxPi+=dU+!*LNS!&aY=*|Gd6`_~q<;vHSV^$A>qcUVq&GkE^Grzn@P#_Ws@LKYx0^f74>pw_ndT>kl8lwDp&V$B%EneA<0A`*1iAAJ+Hx`?p@q-@1L=<W-;{)7S1l&8Gr2VD`Fj_FxZpE%`Dhi-W$t{EEEm{r%nRbu^x+KimHR-ZpDDdF#u6nT}`EjxXQ+vR@1beSMj!;AiOwukYsX-!G55kL$<zBAS19xO(8yUCtNLhlfw|Mbs|NKmGsCIQVAPJ2sW=;2aL{Y?Su>dwut^G`Bx`-kFoGTXVS|uJ)zdQJDTJoi4Ed(By!f(5ztcmY1;yV>TI%X2#mz=ri^*?sVu5o;%-p`yp(nDOi^a;cx?+Av{|7*>cbYZDi4*lTY5ZrTSRP-{kWMhVbQt0dthin?8uUckDiVK6^i)58lA-$GzvlFTbRdKKA)^!iRKV`+p~I8v5My!&i9h>{hu5tjXjsH7<}bPo1Bw&h|Zd3+DC+`DtTDjA_B^hx_~W>yN+vY5n-|-Tk|NJv<Wz4PN;r#u6#N<4AL`y|pLp3HQ*>5t;osxXLe|3=8m^UjN4Y&ilBkd$+0m*J+ag^R6);Cq_6}xD`JG7$a~`;9k8fZOcsNeVFz(>ti~Az_B+BQs%0_PuT<6SfEeo1DQu4+K(OnXx!wY0~HUdWcw-`i2COF{1Z>7&-GP+r}S~qTQ-~rVBGH?*&2iS=5K)$Vq50zvmTe4ssuMXv0?q=Y2%+J-}}IZT44Zt(Pb1MAXzju*u~a&#W6G|xSdn$pl}UlhCnA&CtVCf3<QKThL=Y1Ze-y4{<!WN74S0W(bQM~Z;9qVy%97UqGX<s;o(+W{%8tN131kB013`TM`XwW4Oi*VlYfq-{o^2KKOXzzu_h)JTQ7F39t6{asC;5+T~^M_iZ5=0BSn|b07Lqshnd|~F;F~6$!R|ciT8S;>`o8H=I!0%zeFAD1&k)$(OrEp1dWDj*O%fDO~;~#AJ7g?8$jGK0lH8SKIprSJ-^j-W`I4i8<got<yZy)M-G<VevNJiWgo7T2YvrUbg4|=H@B~>=<t?cL2oYbhDvz2eQ?Wh`e7jacx*4ibLrSfmtXe$0gdlQ2Yo^#>cv$0@bU3(^V9nA@h^ZKD8-G~B^EXuynXT11BK)<ro)yD2`+8)BiT2i^!QmiZiZnvhp+k}B_oQ#f=-)b8BJ4rV+v7um=O<Vb*+!xhutNeKTd;Tw|DGh8)7c%z{q2lzcC+y;wp&xZGHXR%&LutK0P<I67g*BEy6z&sMF5lDmdTK*l|DNOkXQ%b!FQ$C$dyx^q}2oFR%K%5g#8=x`YX|EB-NczcYNLa<4D|VsZ;^9v<$$q^Uq7>gA6wGxYg@d?Q2x_rAClt}D}r&f%ms+%iT^7{nHt4{CG)$VTjTa^xYeL1)0w0a-tzZ~4(NFp~IExojm=s4IY~J05*Zqc-lF0#_24+I%X*kLyBM5i}8?nEOu^unvKK6aEpf7`wNEF#_$JqZ4g@HdOWk8*B8*92s!_w8-u}_FUsp0oP(?r5?)|x~k-$+CejjxWKU4DmSiRj;L*M9fwks*-ZQ8(olAGz3~nMl;P<{#%jFEc4j~bC^aX8uQPLv!vyLH0=)A)?f1N#5iO_dl3c`$o-)gM^(;k{2Ta#HPwSO21p$8DMWbMQT4O?m9j&{3AIG!UQ>Jy=zPCL##JT7TwOi&kZXLVlbz?ESZi9vxw=&p-?GZrkebA!W--;Qo%qB=blpI9R->_4~4wE^qbkjHL$aGH+J!Gm=j$MG>X0{u%F?X*?0ZWhW*ABt%Y@MepoDYgYf^P1`5#(@p?cHF#%a1Lla{#mA*stsf9#>x}rbWhnM?d-Q7JCtv)XbE}*WNhrXO>4%NQR)s&hC#rl)D-$ZntTtIBsOvZ+|O<Q8OVS(P*T6P;4J;--;<irRcZ^9fG5Ae;mJD%s;=o|MSbri`=31mwC=y59<B0rJvuoSmrx*$*=%>D#7BiVFB*M=rc$tw_wGAtu4sGSbn7$C=!x2k15~;j^;fbb828UFh|SU^jxsoSng(WdSv8G&+{a0R|pU`V|kGyqAlNIBKw7nR^(%~t;7drA5;$J9XJ_mcLC9|_5sDnXv<dxE{lka#C}Z9ef4nU)~aR>oa67+5DYM8U>*w}s`=W~lRQQ1*Z~;xyNh6i(jYhxXd{5TEIU(<KC$gd95hK9P_sL=En&=>g+5d<k~i=ESyn^?^d|yfR3M{lHG1wX!)g5g(aZ2R0G;k6&+aIpL<2YDgT9--if(@LJ53Dg82&r3RWkRTP;b4ErtZz>LAu8D(VY4o_r7ryV9pOVNM%P?{$Xz(BXjfHHM~dS+m$q~C*|zVLq^#OFDhVR0q1BP#=~g<mWGE3Yi&jFiPWv1-lvECFVhQo%ub26(asu{CR580-D>G@sN}VRgNnT?fjtE;d<>U>@}8LFRttN8%}~j6mR#=EhX$4^*^-064Jk?LPVi~L-$WG=owOZ6(k!V+@_&J;2f7&lsxxs}SqrUMsst=sgSAfk<J<h$Ew-0{$X)1l4s!hrAxsGwrhSpcv|SZ7yc0AFm~ca0dCJi`S*yO$!5U5m7$amenlDgmm7MS=LkVS<I;ZJ;y>LC@&s4(ob+t4ZjwprgsflNb;sh(}`l(NuSm};D=tm)pf|b!H7g=>BEA&smc9odpFhYu+(XA~2=^Kep9aHf`52hHrA+J&s@Uq;_Vv1MmPlExPzC?`n_GX8vk7iAQVR@xY*%T1IbqOg@EJ<^TX*ijz!~m(+N$lXo**w!Gg8!Vhx266R17)}wP(mye&&peS(*{B=pY&URah&<OZNCANR%-uM2#GEhhH1f8G2=;qdEvQ%a<>e8&J43Rsx1d47e}qHgNG)aqdao4k#IPKTfU9p(k^5Z7niMET;}VM;li_lxJo8>!wwNZkCtcjN;pI`G#T6`<}BjHL6&PerHf8~3HWrVjj6R1)vr`nY_50;lEK(~Y%~KZ?1AJ^olBS`$rix-rezM2BZbw(3{ByDvGw)KLk^1oD{5p1d~!gtz6uT!lDz#ajWQ@<ds$AZ;Ly#tl3)>46t;Zxqf<5ZxdHugav}R#B5)WICBX9(KpciT$cf@j@dwLgmU$T6DI?x<4|G3fp<=yow6fsr@H$vTX(tL1?3P`^n`P}<Pnb`*p>Lil08$G10XLY0ZaFu-i=ly-%Sslv#v_QxV~JV7jA_7El9CAMif>FIbMfeiLSrN&>4+n*j$#-bjJ&yMxC1|a)wD@mY|(g%p4=(O2<iROM*A?8%$s)=N!Y%sE-T`e@dd54j#0FT3SxNlpBKBdiR%<lfD-_4alcFVThbLOUY0F1(0Yo2D1|kuNkSAL1TR(=Q~Mi7x{j<8@mS<Ca_FxEmx@w1m229?U(@@9el=|6&_EwxQi*VEP34ddErsiIeTd-aLvE?znidimBuGm412`R07$ZTL-Ivq$+$5OI@j)4Y&bYbW>$)P*?p{LqF%FYP##T%oM3M+oBg&9KNE?}~sSEj;PyJfWPOgaVv4E9PQDw=ntBPhC<C-vW0s{$@)r)VRDg-Ht#xT<S{3}5T-bgXfsa)LO7$jH?>}WZeNFi!ji>`$uMCXgSy5b}*D~uN^C@AGIR^|-(V`*()HBSh;*0pJ*LCcJEr6|@^y+;;vWnMb-wOnr4@~PsAX`x|33TVVIQg;KC`%C;@r`8a2mQsICLS_{YPsj)t$}Uk5M*L26>4KH0AxgCEb<Z<g<^C38Ep)tmjqZ1p_Lir4)RGknGYN&w1AKzPM~DX@wa@jVpVOMJr$h+qS+5Y$>U(jvPey@gU28~Y)^eCbgrMMd@7=)NpNheC*|w_Su2N1!CY%OfxK5HknBUzd&Yab%h(U?F&<W18s|1ZI|4t&KJ)b^{ke+&DFxfTvl$;KedJk^?fF;kEz7z@ir3HXgkqg>vkE!1NErUahw4H*+BwVA6D`97o^r$mxYW)0@G}VPUbyk%}@f)J2K=aV3#}NOla}v2wDRXfoE^mcKjj#qfCMi{+vnZIdm&B_jOi=1CWUw{Xr_UE6T@mMA(S{vzpD_QP#Uf+66~YH*|IJ4Lv@ma8tQIedsCy#<>6^8xZsLlgR1`WpTLXwzezRKFxL4kUW(;JB93#^jln)bjt5Q-~Yh*1dejE7im#y!<_<FuO&sIP?b)a9qUz$j*EkqT%>X!mPs|_Q<l+`TO-0NExGY85cR%M2|RVFsY5%bs!%pEI_k5a8nFA<U=HFEIzrlx*KovSU$`XJ%))$`E4e3O|NlPb)eer&e-5bHZWEOc>qiA~RPku*-hKy4Kg_RZ$4PfJ7O)E9<abiJwCe#Kh@gAQ*2g|o{#tk!=~Ge|<G$E*Q?7Z~>FqzXKFIvuf~jY6(NP&F1xRww|)BIVTcTEm@8UTjDIE{uS`G3VpPfZ`5>iXJ-NMH_8us7@z?XXm&%Ad1IZ(159Ns|Ry|F9M<c?;>L9xIrfkSfX8#xJ`r7wT}?XluDZAX1huvCLglF7A{Mp)Qi9?27#wKPaEE6m3;b6igCMg4vYOgi(?q$Ke5&|&Tm@z56Qs1GSVq~`QX!e$3{RN6{>iYl1~B_B@3Wo_MPbBrKZfa!?DayD9QJ!lxY_+`kN<%Crtxw)FDi*TVJv4RjJ}VY=xGIE|J3v?e_EqJxfsQW!~rm1+^Far9#&B;<BM%C|-b?zN5zCW-A^x$I7oQEE^IBP6(1(B5NXIAj%~Qo}Z#Cnf(!;`Q*qM+;WZOog6-t7f_d{T$G$|Ev+Mw5AGlgqn=zy1+8phMq=8e;)Y}T(Ov=K)|9X+#_Okn)oORNI6tGq63B&|{lgS#m;FK$a^WN;)rM!h)EDAqJ^kUi#O3~977(X&`0WECj>J(r)>YGSc|&>hQ|Uj@#M^H*7GI;ykBQ3HorE#*_{@*i1LnsG((A<xx>%upd_HZ(VBANU-&SmDkeFj<EFZZ4Y3Rksbk?HuddM5y*z}q$1mIa@yPBJ#6seaY@v2-Ssz4@3g%qMOc){VqkxY%_2+x_)giL9m8}<~$l9m6gEFe}7k~mdKi48|ONr3^VA|!EtDgl1UjJ0DT1<hOy6@wNBWkGjRQMF`pMeK~i!qrf)7A4m-6^7{;Fh;f{(ytNZ(?tf_!6#uY$cDzz^%3JlWJ+D&Emv$TOe3XS4bgHtCign>_f}uexdX)JuSzih<XM?xKrW&YlmZ%u)5W(t$q^Or5Z+q)mpmv};Bd+gWA=B!!N#b6e;)RDiMz~B1qZMk$X6qGaklmCQ^9vl&F9D=$vOs?<lKT)u)vZxOhfG=qh9jHvV~U%-4!?cuzkAs#FIz|&zQ!Jr_o8ZK2T3c+~Km2FcNnOH4(USlf8U!u|x&>50u_LCaOtb0#ytkV6HHbV@O=nIHTEdCQ?Jx$P{{9Pd&lwhC{M2s4z4SBY?vwd7#~jlmDD|@VaZxlBlj!Y9%{>$IRMQHRdj<3L?d9#bLFkN(xzP<PI%WBV2`aZ;d!W>hWssCon7%vkJcOYo{*pRqD`Xoo$|$@myFvf{oJIwR)_OVa)ru>YXNu4??;Zu`TkZUs+==CDK@3=%J={>?0y408|WGxrX)|K%a<j0kLn#gT0okxOOSouisD(&hp!l=qI&lM9nn;(;Jd-V}Ub90GF0}EN0!vY6|E2fA3eDKZUmcMy*pYSV_T3#VMrT%gS=JU!A1@G93o$Z>0cjQuB0Ri2(E|La$0YSrUYtp2)YQQUc7&z$0r+f_U3nIW!`xM6A0xl$x(pCBRu(iFKS$#PqY}+m&h;@Hz@odJwIWuEJW^Mh=a*AP$u4nwAMW!d^yd#dJG83-uV8K?GARq-Q~};Jmxq0^iv<WksF8m}qq@Y(xbPg?g2y=s`!nR`qG5s0Ypcq4_4zwyb4jDBw@)nro$D8R|V)fxpKoZ%=-j0@G3bV0Z@|o(%0ulT2~qnaVptV~RpS&kWY@sRmN*)6kkF-@;a6buVcBO5Rrl3qOm$(T7&f{Pq>ewAx76!5da`MUAhR_OriYzKv=b$R^^%Xq9PoF&qrbC~GoO0Nqa+u?R2#B|F^kfDOl4ruCGFFH2)tS*oUW_F2~9Fu<IjU$@4>tQ5<X=L#9=#u7yn`3ms?3~PY_Y=Ynjt2CZ~ZmxGBY)-O@Sq*woo>GG#%}##dYb=+-EV9$IP}r4I&or<p?`u4H%NkQaVn?dn4{l${oxJK>hm`cCnt)EOp4bbr;Aq_xbvw~#$RL@{wM8dW%9z*OLtkY{Ii#hGx9GTzz#ls7)l;k|!EvLIGu4_VzH=j-vpg4z=+e-ty0lnrVY>nshUS21iHdnyq(1P_O5-bLK9Y%H_uLSwa*{c`wz)&cn~DxNULj;l3r|?5<A7UoJh7FQP{kxNk}Wb*JElNInz%-dLZ?EtWCkc5)p?d$6)CM2PTKAA(z_VM54(63L2*k~tmb>KOfy&&2OE8Qxti4A8^96E@PQO-GJHZqo0-91ltwY6oNISA-CQ-@t5%z~({R;vPSp4uOJI#3dJSfRTS{W&cDlyL(t==iAzDXAyt_sVQe`GQnaGvXI43nx7{CqV;82p*k+oG(hhbO|Lo}m&CXJoy1Dcn{MJ4$p>JFtYwC>?tRgiNh7mzkDQFirPPrxCX01I0|6XzH~M|II&)gU6K7Aq|oye&&m_p;v;qo_NFfOs-lAv|&MRA4r9O^9eaP}C9~V}*1mVBM<+AZ1bnhika6Rp;u`g1j=Vb5RMHk!XtfLvu;hJWDOfNJXEb`@j&GIyH(!urD?XTlULyEN$gXX&ja)${Ww!{8?jlU@$$rp|*3w0=h}rqMiAoI+!CVqPYw-MF65F-|v;!7g0!!(I7*PZp&m1p{{~a(Z|o?O3l+cVUUIL1DjIHg;DKR2wSa=r|orsF_8=q5f>U)7feZ_loh><x)(k5!B9qgw~m)n+^nM{wxWv!7mJ^^s=#M@>Zq1$#?*0J-(;Q02Fr>tg(&30NK}OPjtXlgdb{$vD`cOs6rgC|0?DN{FR~0q2Lcz0t0Jq|u%t%CQazUm3VMm)3W;q|;;_M^BB|Rf*V1MY^Nv9IGMwBSBWrTwiIQc7oQW1!Wi~H`*(6z&Ty%}mOZ54!94yvsTBY#hLY!dRI=V(f9Mx7b?7}F6#wS<Jwa%c7Q;~)l5WytT@P=fkGple{oJK=re3}S}MWhawp+Z1P;y-;Neu*sAb*X5SPy!^c7+>eBMtiYOW6UI5Nwo@w2ryndo6Rfntb7PpI_Xlh>W1#c5UTSe(9XU3Z)tyTjQ(CY)ejXlvTcg3yfN*~cj@WoYvWEMA&U54f$BR#U95%i{&8xiji-qn$VcQvu`t8(<UFxdA>Dz^PD>f9)MWd%d0|J8tN$7tv9vZhx`C*EolF9gND-6;rYZy{EN>e~sHNQME1%L)sN~X^4B`RAu?kXEq`8>{vb8dzLNQs(<68|PkI}EvA$4g>m!l|S0Xl-IgbC-+!ERT5tg(dUlFCm>h2s5i2IG#oiZOlxK3_5Ys-HQQskc*oDV=Czq<mHd-NcIbq;*z>k-=G@n#>s`nT7RVlsa^!#Pyq+t^$WfXCicS){;a25enZOSIJ0WHtk$`(pVN9kZ!P^DOJg80L$Tmg4FJ9f4&_kV8pd`;QJVA8lQ<WD>W@Ux<W&BzR!o><3q1{zCP1R$CE3qUF1~suTV<3*u|Hjl=*l-oD3YVZ7yreOPl6~29(4&`;zIfRl&;3v8=aCh4r@hp4e@mmls{_?`OSnD%p`MSSG|Jm+y9@vXw`pydAHO<=0}ojV)uT39T!hFDERefT3Qo-Bw1qE&0I`WacmwnX0dmfRs~p#0qfDvX#p|a5@AZoiPrJ=q$rE8Dt;mRKWv#u(1l3r7Ao*yN(sT6xnqu38Y7bDB?XV0Z}KPXF>oC=UJ!XI)yF9(0PnqtYi{W2h;vE4qli~^~yko63V1)x){|{x{Glx1jvZ+>{8`QG;2YQ;+U>p13G1q&`jqN%qmU%4iDZw#BW(r#LuX$G3=i_Vy4<&BoL-6vh7hqHxgyOLfQdA<cD4}I=zl&WQ!F`8MHU9gDaVLx2*8))H)+nRhD{Zy@gq8Y)LLf-y&b}`kvr=w)UB|YRiCbPR>hsT?FP-t7)BJME0m)YBcf{sloIHU#{Nm%lbl4dabJ5h=nG=Q@X8;19<Z2#Yk^~b@$D}ohPq5>I;x;1c(h!_4QX!nbUEO-tr4KPhJWzt2bIznoBgSOro7qr&O#Bprl9{;9G?PN$QYht0j%-bJ4EOYNN8{Dv4!Fqen2A*0(iP?pZ+>8aFnaQl$N^-IN-;(pq@Tw@9h84A>L%3-q)=#(flYl8&(KsQM#SCs!pM(sgE*<IF(T3e7GbIjOEDr(ALM^w9H7t-*_O${2`cio6B2>Wf;qrm<Jhww<<ImlWkwk(GgPrPM81HMG@IY+k&!#>MrbhQPoOAwQuhQC#PO5+56MsfFw|Rb<qgCl$ruMrdbgDuQYSpb!tSKsiAaomEdrb510>8CBKtb?lAyGm79^DHgj0+j6d9nTfpgPKI3FRfEWH#JKCYe%XMCRu1zz5F+?)s#3O;wiDH<^kvT!5V@^YdO|*!fT%E|skiLprciTIE%jDldmV-{jz#zq^X59`&!aY}?5%`G((B)~q%4Miuw+5CcmdpesS=At!1C4sIAK(E^$IwD)knPojRb##C!?GutajysL{j?5R#{UG1Tz+NjV4oZD7t7>c}cD`^^6?JBG0irZ)9=GOD)kwQsmG}rBh3)bDks_F}an(s-jw;I(s*jWf^7>si`c*Z6#`tRb#d7GG$uVLER@IJIR)8vCh#})Zlp{(1LBJ2%I?Wk5J~}#LVy2BYS?-Wu0V0(g~UDK1s)tHf>##XVtZ3DSXrJRjsZFq2qMdseYc|g*?HX_#8W~*kAyB2O$p6&Y%n{EX~baryybUEfMdZj2BC%>@9VA9%qRSC)QCeV7$XwDyBlwo_6dyBEKxXop0r2E=psh#J<?yc;wqyB(IL5ZpidAeO(_rUa}go`v*gi(_Ob+Qqj8n&QefcZW^=c9r+<FU2~ec0KZJ?Ff8)NDhhE0x<UIP&F=&RHdZi2k2SLoRY5nI!d76H6gguhvRogTu0~(b`D2;Vx?12M2a$o63YcRa#?%G3W1)YBWiEP-nY7|haSAkYn;IId!6_(DR?z#ifP78xB~2K-H8?V9j#!#brsTP>eq4u@lk`K(V7{o@x0!7?P)%Pgb)X^rc8q^j%A{dj$q9Xdh~EhVQ432{t#S3FM+CEVtd^$$-C;_e=0Nl070SnhK1JHl3zN0Zpl2l=QiVLF^TyGE$^`aI$F$XOo1hnZ(cJQ)hts50c2MrCUd5*iC%3Cx#LZ4^^k$Zla-GnLR2YJ!BPW_8oL)+acHLQRdV1r^61L}aK*^g_F(cl;gW*Me6^m2F;UqFLC8}qOr5|SR4d3Qq{R2TazSJWfO1ciZvm2E&td`t(8Ns{Y0$C!s?czJc)(+O$1zwV;QGAm!Zgj#QY><LTr2=R)(<Q}EpsOl+M+~X7lzS}YHzHu`mS$yg_GaD4zOoK{Q$z;~L!_{LTY}V-7&N^CbwjH8wdtL1V3uc0v3nv9xhxD^K~D~JV8ujF+vQBQ44Iq$gs+4UEZRpRL)fUtw<tlU)UcyjFVmLKXa?S#ET1X~gL!!X8JqKjzw>>xbc&a1DMm8?PmDI3wxL!Vh9-ka2bTwAT^p7Y%z`JlVKS(#fTh(?!j1=m3j_(I!gTXWELIb>VgtJ3owG-CBjY+H`UTf%qANzlWfGls1rv8maVo$P5sCEkxH-%=;w(`?Buda^U}Aib7Md@Rt-zWfoXMs-BRo^$li%7EWcg&m5*{##xE~>3u$FQ*PM?hFZsf(V)M_c4>h0?k1Z~NDo0?9Nx6F=;D8+JFOKi_)pv;gu0_`LmHG{ODv9;{WB#rEAQK-L$bjZ9Wu%?+oN)ZJCRJt%!OuJRRima(2RRvTaPFE*mj&Lo1lvm~xm0~yvM%z%D%tDG3X81=EdN%6QZQ>Hjd9_}#WSgE5m##H-JZ%WsG9RRNbx~KFYIn7a4W1TIOi0&N#xs?tN~K|y9xf_dLB$cPLAigf7v!WH&{KKi^<J!?0rLwi{GFP>+^Kgf;0kV{#ulj+*<mG4jO4ATrC`1_BRff)j5m$dJa!#zjJ+ZjP#rUVi9niE0650t{WfCm5+)iW!p5xl8Z68<tMk@5$$N*pdC>vUsHG>g|6%^O-Z&q7SI>w3DIMFMF>^_M?rT~qZ|Wwa`X?>zqED45VPf*z54l}+x(7uyglN2y+L9ErOmC~O+@A9CWUMJ+G|`#QUd^tQB>t4KTXRJoYPV#Hh@8fr8l1UNBFbgzx)15l!(&}cIZf(<>hA^@TtIA9>X8a$E^<e)J9cny@sN|(R*n?%I6ksQXE)AG2(%poW5xp{lsA!ARvg&&O;VM$-kuU|N>aF{tEV6wl}MHPbxA=_>EWjo={ap*bxK};U*lGi*S>jtcz^iTmv1H?cwfQKp1)%0;s%Fa&`)Z2UE7J>hux27-=y(D(6+;nVPgbu4d|!8Km8w;Fh$h')).decode('utf-8'))
_ACTIONS_8C6S_3Q = json.loads(zlib.decompress(base64.b85decode('c-rk<U2j`ia{MoP)`Lk=UwPBm+}K#n$dK(2n}IMGAR7b-HV>1$1^eIQSR#3OPjz)wpF`SS_{juC-+R7KcXf63um5}Y@4x@?x4-^z_D{c_{q*VG{hQz3-+lP_>2ZDbbbj_9zyH_2{rBg;eE#^i-~af}zy8nX&%d6%efQ<B+J~P${pGj2U*7$EcYk(%_WEIScD`)B{_u9aen0uchxPi+=dU+!*LNS!&aY=*|Gd6`_~q<;vHSV^$A>qcUVq&GkE^Grzn@P#_Ws@LKYx0^f74>pw_ndT>kl8lwDp&V$B%EneA<0A`*1iAAJ+Hx`?p@q-@1L=<W-;{)7S1l&8Gr2VD`Fj_FxZpE%`Dhi-W$t{EEEm{r%nRbu^x+KimHR-ZpDDdF#u6nT}`EjxXQ+vR@1beSMj!;AiOwukYsX-!G55kL$<zBAS19xO(8yUCtNLhlfw|Mbs|NKmGsCIQVAPJ2sW=;2aL{Y?Su>dwut^G`Bx`-kFoGTXVS|uJ)zdQJDTJoi4Ed(By!f(5ztcmY1;yV>TI%X2#mz=ri^*?sVu5o;%-p`yp(nDOi^a;cx?+Av{|7*>cbYZDi4*lTY5ZrTSRP-{kWMhVbQt0dthin?8uUckDiVK6^i)58lA-$GzvlFTbRdKKA)^!iRKV`+p~I8v5My!&i9h>{hu5tjXjsH7<}bPo1Bw&h|Zd3+DC+`DtTDjA_B^hx_~W>yN+vY5n-|-Tk|NJv<Wz4PN;r#u6#N<4AL`y|pLp3HQ*>5t;osxXLe|3=8m^UjN4Y&ilBkd$+0m*J+ag^R6);Cq_6}xD`JG7$a~`;9k8fZOcsNeVFz(>ti~Az_B+BQs%0_PuT<6SfEeo1DQu4+K(OnXx!wY0~HUdWcw-`i2COF{1Z>7&-GP+r}S~qTQ-~rVBGH?*&2iS=5K)$Vq50zvmTe4ssuMXv0?q=Y2%+J-}}IZT44Zt(Pb1MAXzju*u~a&#W6G|xSdn$pl}UlhCnA&CtVCf3<QKThL=Y1Ze-y4{<!WN74S0W(bQM~Z;9qVy%97UqGX<s;o(+W{%8tN131kB013`TM`XwW4Oi*VlYfq-{o^2KKOXzzu_h)JTQ7F39t6{asC;5+T~^M_iZ5=0BSn|b07Lqshnd|~F;F~6$!R|ciT8S;>`o8H=I!0%zeFAD1&k)$(OrEp1dWDj*O%fDO~;~#AJ7g?8$jGK0lH8SKIprSJ-^j-W`I4i8<got<yZy)M-G<VevNJiWgo7T2YvrUbg4|=H@B~>=<t?cL2oYbhDvz2eQ?Wh`e7jacx*4ibLrSfmtXe$0gdlQ2Yo^#>cv$0@bU3(^V9nA@h^ZKD8-G~B^EXuynXT11BK)<ro)yD2`+8)BiT2i^!QmiZiZnvhp+k}B_oQ#f=-)b8BJ4rV+v7um=O<Vb*+!xhutNeKTd;Tw|DGh8)7c%z{q2lzcC+y;wp&xZGHXR%&LutK0P<I67g*BEy6z&sMF5lDmdTK*l|DNOkXQ%b!FQ$C$dyx^q}2oFR%K%5g#8=x`YX|EB-NczcYNLa<4D|VsZ;^9v<$$q^Uq7>gA6wGxYg@d?Q2x_rAClt}D}r&f%ms+%iT^7{nHt4{CG)$VTjTa^xYeL1)0w0a-tzZ~4(NFp~IExojm=s4IY~J05*Zqc-lF0#_24+I%X*kLyBM5i}8?nEOu^unvKK6aEpf7`wNEF#_$JqZ4g@HdOWk8*B8*92s!_w8-u}_FUsp0oP(?r5?)|x~k-$+CejjxWKU4DmSiRj;L*M9fwks*-ZQ8(olAGz3~nMl;P<{#%jFEc4j~bC^aX8uQPLv!vyLH0=)A)?f1N#5iO_dl3c`$o-)gM^(;k{2Ta#HPwSO21p$8DMWbMQT4O?m9j&{3AIG!UQ>Jy=zPCL##JT7TwOi&kZXLVlbz?ESZi9vxw=&p-?GZrkebA!W--;Qo%qB=blpI9R->_4~4wE^qbkjHL$aGH+J!Gm=j$MG>X0{u%F?X*?0ZWhW*ABt%Y@MepoDYgYf^P1`5#(@p?cHF#%a1Lla{#mA*stsf9#>x}rbWhnM?d-Q7JCtv)XbE}*WNhrXO>4%NQR)s&hC#rl)D-$ZntTtIBsOvZ+|O<Q8OVS(P*T6P;4J;--;<irRcZ^9fG5Ae;mJD%s;=o|MSbri`=31mwC=y59<B0rJvuoSmrx*$*=%>D#7BiVFB*M=rc$tw_wGAtu4sGSbn7$C=!x2k15~;j^;fbb828UFh|SU^jxsoSng(WdSv8G&+{a0R|pU`V|kGyqAlNIBKw7nR^(%~t;7drA5;$J9XJ_mcLC9|_5sDnXv<dxE{lka#C}Z9ef4nU)~aR>oa67+5DYM8U>*w}s`=W~lRQQ1*Z~;xyNh6i(jYhxXd{5TEIU(<KC$gd95hK9P_sL=En&=>g+5d<k~i=ESyn^?^d|yfR3M{lHG1wX!)g5g(aZ2R0G;k6&+aIpL<2YDgT9--if(@LJ53Dg82&r3RWkRTP;b4ErtZz>LAu8D(VY4o_r7ryV9pOVNM%P?{$Xz(BXjfHHM~dS+m$q~C*|zVLq^#OFDhVR0q1BP#=~g<mWGE3Yi&jFiPWv1-lvECFVhQo%ub26(asu{CR580-D>G@sN}VRgNnT?fjtE;d<>U>@}8LFRttN8%}~j6mR#=EhX$4^*^-064Jk?LPVi~L-$WG=owOZ6(k!V+@_&J;2f7&lsxxs}SqrUMsst=sgSAfk<J<h$Ew-0{$X)1l4s!hrAxsGwrhSpcv|SZ7yc0AFm~ca0dCJi`S*yO$!5U5m7$amenlDgmm7MS=LkVS<I;ZJ;y>LC@&s4(ob+t4ZjwprgsflNb;sh(}`l(NuSm};D=tm)pf|b!H7g=>BEA&smc9odpFhYu+(XA~2=^Kep9aHf`52hHrA+J&s@Uq;_Vv1MmPlExPzC?`n_GX8vk7iAQVR@xY*%T1IbqOg@EJ<^TX*ijz!~m(+N$lXo**w!Gg8!Vhx266R17)}wP(mye&&peS(*{B=pY&URah&<OZNCANR%-uM2#GEhhH1f8G2=;qdEvQ%a<>e8&J43Rsx1d47e}qHgNG)aqdao4k#IPKTfU9p(k^5Z7niMET;}VM;li_lxJo8>!wwNZkCtcjN;pI`G#T6`<}BjHL6&PerHf8~3HWrVjj6R1)vr`nY_50;lEK(~Y%~KZ?1AJ^olBS`$rix-rezM2BZbw(3{ByDvGw)KLk^1oD{5p1d~!gtz6uT!S}FKh9A!|-_OhH-!NHqvCB-7DDQtP?M<;9SvjZsMU5u3r8Q2nu!;mQf-lqWMFyui_7N3q(MZQR8V{|8uc+Wjh|7dvfdI4!=+1X)su#nPj6e8Iz%Y-+}+Pj`WpKw#(JXL_C6#4`1FbUyuj(QhU12LGDY;KKD5Ru3d^MD!EfVU(i6VMgkm_+8{(UFD5SU$KEPPIOY0dX+)=Ca|g{P=a#rgO1n<7s<x_aLLD_lq3u16DGF-qj{yJFB{^h=ax#w9-08)*^a{;o*N??9wK#SU{mp0L8`qF5PfRSE!6xw$MQ9DH@`b*r=upQK%5SSXoT%ZyXUjvP#5bkqgS9zYbg~N(EJ}ZWn(|?-Tmfu$4nYet=UY0<$%}Lprn+xX+a$f}an$rG~3pNU)F~F5M61bckY%1Y>qzPTO;nU^mAHX8=m$W__>g+C;m131!MSoEjNnF&Pm_N=%J1LqZ~LjIyRP<mW*3t2R4XBf7@|Rz^jiB}1?(>S>I1!ej~zJW$p#zJ01tr6?Z5==1Zh1WkA&*+3_CaerfwbTzP}<+LJ&vSls07LX8~FXrlslenyKU#PU8l+jq3Gvtq@ResfsA?#Y$rjdp(Gy0XHT~qZQS@e~8>CD%1Sz^nliYulCi3L%h5yME`4N&+m@q3-BL(Ex9{W%GnRXjW)H(V&dL_rwwJJF>JR-%R|(YDt;&v2FdTZpyL@$xm|-%;9Ip5{?YRw&UVlsFIY2?ie_MugNp*N=WqaK4^WBB*D*LX@lT#o0a?6{2;mA+1@<W)2a8g4?}!19yKa2H0iWs)D;p*%6t98i3(C>Hc7TcbhnKR;zLbCH_JuIMc2YG^+eNiI(<!`Yb}C>W#r^*MwAZQcUVSxS0f&P-FU1B<+`$0#fBJXtO=edi%Ew4lxpV3Tl&ZjWV``ombMM)2Qk4^H0)L7v|JiRUXxEh_V9BNTVJ@{IkwU<dUV##gSOO6+$(_8t9m$RE5s+V9H(+ua+=XslTMb)>xlDUx;)??0ZEecF29g{CgIQj0sl=ADI0&9|6$9ym{GLyeOjXjR>S~)~e2lE0a>u>g;R{AX@p&YF*=Ap%a=rkfn5tOmI*>O!%!zab>NMwW#=Q;JaV8zWd_q`R+Vx0qxX*e))cBBHgwSRp_c@3jC}#j0lrgvsiPlZ(YnED2G^;8S2)X*c3<1V=pjwtUNwS-7>vcNQ%_R!RMQr4kC4~wlwR5)W=uPL;LbgW@1e0F?ag0+3G{A@A$CLW!@z|J<CYaI0XY$R!G=4o3}ny4Utn{7;@3|rn>tTZw(AOyag1_F6*#b2}aE!37sCZ8U$Wo*rStb@#IN$#DX?Tx(-3rSSVSc02IrWQ_pJ+cQSdg9sRp70{+IFj~fGuI}j>*=y(@xw5g#woeZ9x<K}=U9&bSnrpB!v%muy(B=^6Ih^6BOoit#Hc17Yg4NBWSLM&4%ZkC(vDv6kU$Oc=uERCWs0<RbZp6Wbpc%N0W>N_dM?aDbU`}Zu9VT}L8TGu$gX$?Rm2=mHFr|ji}Pv;#Q0eMuY;!#RI30RaYfQH$3qKlWBGS?2rGC!dt-=|WrUBu{bo(!He4YW~*Ftu)d?J{9|B(9cLltHW$FH4oZ>Qfr@EJ3Z8d7}@s)L!tH3R&BW%Z7fTcmZlkj~a`ct$5fRE5Ej|Y)Bk9AxLV8tci$$D4ZyGeu~0m_D6i?lOtzv%QcpFa`;eQKwYSEQF6Yuw2nkbxPvf^dU7F^wz4G}iD{F{9FFNndj*JFQ^Kkkub&21tKHG!{EQAuAQy7>57VYy_6tqOC6ts@8=mn}Ux=6W^oQqSm-~NNK%COyw-1Oo5=ZS=S9Qnb4du~KB>+JaZ@<-8e2q3gCMsWd62`>iGe24nm>(xdua`CGVukwg`Lq>-aUW%VTd}D@l8&9ReBl14p%)|5S&P!^A#ZeJ(`&X6fM=2IYHo^Bu3n17tAdTF7MUCsQi#Uj1&0eqGBu7PJZFj%GNplT*i(>AR{pcHfLLWn;#4KYHXP|B1qPs+ki`9|1o$O0){co3lyfyy3|btN1>H$S)so2-u`>z_S3|*Clw8wP7^Y*u7}=6szebQx7a3>=pM<p_8yZK~M~oAZDRqIjT(Pk*jg)dVM9b}%-0R5STYWv}4iKBaD#ZYhXJw87xrjzk3TPZom*4IrM^wB+cx&lj@}OLS!znwA+1~{R8>9aHdD!D6?lL<S9KdoQUya<w+19sD1>ZF_pCgAP>lj>;a|>3%0!!X74YiAmddVBh7G526SKREw_UYaePa+*WV;Vc2Mkm!uK|LXHhs#33NZcjVMBvIz_VU5SA{FRAP<r>6s3w64R55^nxxzq>A#qLPjAqA~NDWaVQ|NI$l?AUG4#~ox!q7a701l(%fp#lS{&U{J>#jLVqPkMCmFxf>Giz7Xn7gDZh!nFGht--YDP*mYJG4}da23+MHR1rN$E&%Yz_3irDtIoYGI5EoQim?<Z1c2?=fdg{Y?RKf)nkPWW8TMA?=(q#5YoMfZIL(q${KSik;dvm4>hf09}zhLpkmO<HMHLV`b2ySh<!UA?6q9QRZPi#{f2UImfwy<KdDV4YOV>G-jIYF3!E_mxU}SBG3!QFQ#jZEd%xQJDYX4JYMp|?N(xpgP9gPPR+gjv>MRA2=`c`#D+Oqi+NT3c1fWk5dR5xVk|5;tM7}MR5@22i9$8}&#M{=&p%GanV&%=D)O@8X0nW-wtmAwlrk^d}u2j2#*HMttgJ`{U71p{oa%jW_aiCn+v`p9$_A*jyr`zdSsK>|*BA99+Jqv;b=iSv7_|C>DE9(5kM5|+ABPwtx)T=Z_4?6m_s!t<DJ!tL^%{PIzWi2B^0e@Q8Tq_OBQ18JC{5?*2d-Bs1n2zcP!#n8kWN25KWQr5dRNfI9Qxpn%X0U!wHIQnbhSn_k7Pb<rdqL}0^1dQi_*wjoKD2V?x35U1)keY&-msD@YJA1CpZyi{ZB)xZHW4RAt4ynl;b2%sS(A|h=zhwGMSuY)+2Mu<Y&gy`t*1nMSsKgAQZ=o!&$14O0p|Stx-}MNrC6psSI9^=mMEIYSBMW_SPKkb69h+CrSSxGbG-{;bCOleYS4@Flo|wScJd2fW4RP&k)5W6!mgZprh!d)U*pMJ)|dhkJ5uF-aQjN`<W=7~q@*v^1axxs#9ojEN9(4j+lf9y2FY};EjpP}#=Pbp`YKDxAuVORMaOjn{?K8so?<--jvIxXsn#^{og3kt<+)fymxfl=rNwFs+ZDJlGzUaWRLsjF^?`?08ecK<kxUG`=Y~*~lg#0@%^f=4RCK`c3L#rsc)~gz2i%h5iLJDRDkhPUY>}DTF$F5p#5HOZIu)uVGeGI6&a>32NNKfj(r%ZR-o+q(*u|>|id(W`HQ#$>n!&0#*yz*C)uaaB0FGFO52RR=;S(C#%nbgbG>ReRT)V63=Bnvlwc50uhO4G?qQ>V~0&D!xYcLbsQW7J#(=|Sp76h{k(K<Tf-8EW}Dl_59M6R61IjM=l0B#ruhmy38tgVVV48w{Tq8a5gY3x)V(7ZG*D#<5NcPMqCbr0vNf}A_KfV6puva8>E0uIpxSl9}hIL8P&s*Coj1`#o}SZT@NZCQf4m;Ig?Mcp|B#FNPi;faf<0<)QGLPXnvqL%0wE2KjK>s~zoDU%{NT*G~>I#-t#<dtcii%P(ZL{rQknoFwYS!zi}D*6=N2Zq4ZsZk_?eX&{CvR|HKX)9++<FG_g-gxfj&l;-(gX!T7wVfLl&`rt~?aUX|!5m2u&1Ik|0uVj<ey_y7h(c<N1{rd6TPAA=brp<?K7JNgYM#~!gDjLE*pyN(jB2++*lKk=ZLb52iDZC?xX`$|U`i6DtmtLbz38bAhBD&2b-bM7W*sH56<s8_Sp2kA1wPYLN3~otrjFbCChJ5tSXP87L?IVOq9VL^R9G|7+m+W{A^VJ_07d&2NG`2;k!3JC5V%lW6<NiGB{eFR>bXo%&`ShYNNkG|hYcPTN!@0-mNtu+cLd6p;pE;JS(6)2lq@UcOtiQvvw11ZCdsPgqHBy^qR)5bV6kS?DupK(;so2)(KQ<4sJ4<}7e*O0KDlbHbp~aeiZslC2quY!HzYfqS%tgeG#Vn~(?m!tB6YY76#`Nc|LGI)OJu38OGTrE5+HfS_&Q%T+KYV}V<y>3s#P#VfbrVdY+i|H<wLmANtdEkH*_zCP@N}%cJ9@GOZ$6c^!LK4eyFICZBuOJjcISbOHVgn8+RHBQN;fWRNoQmVl9mKk5e;kJWcFCJ|ZWIg&CG7=ZU2X=?-jmTFO|ZCfm2o3p;{b{ny}#rM1b?4Mg?pWD=M}il8hoRUtTGdD}ojE#+2U`IL@AC6~rz5Dy@ZRgkJ8&CMi`t(6fKipg3Y-)azfjDD34sY_eB97P!m&=E`}OgM)QcDw3hjU_CXRDMb-6z_*K7<bH7jPVQb`HJaR{mijUy`Aby=|meN<+Cd2CRV&Bt+OhO49)`8WX>qbEUf>c)S)XSuHV#j6*x3H6QP^4mK^$zQ26e+N=6E^Y3I_D#<J*ubc6LwsY+G@SPmByq;_xn^X)(ZBd)Cj-^WnX_)L^pscG5K6&kAZeLnmiA9~gE^_f;Wo?L0|BB!E%g;K)BF1`$<%*O-bWZ-yhb6Hzn+B7#bpd`lGmrRGP3RYf@WxZW0thdGY#BKw<yy$9wKkJQC$&OsXG9fOxe77T&tvnj#?Ra%8zZT<dY#B>UXkGDqIbkUU4E2ibwld0X$q$wwGl!wbRDF#Eq@1cFR)A}ktz7nj(;@iijB!{*XBn=^Ap1b43Le;lja9HLRpH6mb*$*6$gWdKAU!HX5$|CMh&uT^69Q;B&pH*?DQq!@&SUIiC6kaknD(b}@WOnmR|YbaP$q5D#i*XrU5s-fKt_aTmnv7HSqpL$$8_}?&?$?AW;&N(R%zmQc<}Zie#?>~enxGLVgKY2Gu8GYfiPW>ZI2SVktp*O(hdkBKlGZ>>2)+CTdY{hpuKS&T*<t<WrcUA))}FyveY~4EzDYDOL8gt7Ws<T_XO9owa=_oTLyG<a$dsgA~2^~P3r_BvPT6|qmi#j4W>8va`kRs))#`(YgOe&EHnY0(rslNz>`NWMtTdZyKffmJbB$wUw~vIKx}xbufKxIoQ`w!mS4De@=}0Vz0s=DT%uuR677^arDAOWB}K{r-zpSHQin8KEonrbi*|KZ8<j0rNi16$J%Y)!zOAWp&kDNGxUu1sBJFqWrqtM#*1}`HMM|Azz@C_2pr-{g?xUcSbcAI`)gP%kxhm<9t~0Y7X9luXXm<I?Np&?j<%+APhn{b04PKN}#y~7n<SnRGU(~`ijlF`l?X=~(q$r<?tPF%JrEbZpp{<r;^WwENF0L0f1O|o(`3X&l;yM?U_}HLJEo8T;BBS0ssVD|FLOV-S5mYMxg?NYs$_b+Ata?J4b0X2rsH&E)V{f#dQ3TgYvDhuxmU9iuOys3^GUV#68bp30#$Ctt%LYWWa+ueF5W#m-m9nL@ov2QwFMFnd$Zf6C6Y{|XM1>hmy=5mig_@IUskZ{#>oAmYEW($VH`ghD9<@njZzVL6UjMEoWikANB@3#>3*hEUl~^nSmbVVT38Sj3SHStJKI#={B={RV8Rax#wJR4SlF~=E%9?5*n6aR1G?|J+(M7AuOLC>DXXHp0d5-0IBa2gBYKbnAB8Ofoomx_z^CZcL$*mMt71aXO*}JJM%P@;bO=T%=D^Yu_8mn!WDbu<R>OKkCNw#E*b&j^82G0|L7Hm63;KXTvgfb5&W`3_8+4G|=>m(bJPRL~UNjjFaY3rIitFA3e;hT1^YIQ{j9jCia_45QT<O%M?=h$(@1_R(b2yu9J24z@bX>R5^1qq{XiFp5HyjVJAZ>iJsI7@6ev5s;9;~maYF%^pTv}4y1`DN+td@Cn&Q5qv9_Qn3jBj3g%d36+ZL#ChU>-yO7lGTXaKNy0X?z-)giq_?KmV)|n)0j=~$PZ!Zn$y$;_+?UuVUa&pQHU$h4cZTBekUNXv4SajteJhN3cAS@wgS7P$Qdh<<@(5UHTr_iAIp^1)dB}Ohzzt;z#Q{1rY^W03;i=JbJ26mq!oXPQ=pOC)X-oJPC<FHg5IA6<ZFU2X~N*G!I4RG#L{#!CC`QR<2tOIq#t4i^F`IZ&1}PgYWixa0}bi7WBjX9CJp0CPUs6n{7x8%T3DiLjjJa;BABgXwLAsr4pZ_p2bw3ZP(B{?Dbj{sn5=aMJuB&uD&#4hH;xWeCa`BZrmcqC1ijFU=9U*doF=WZgK}T>Dn4B}xn12NZgy&;H?x$K>x53E!Vn}KInf;9^ioQ+>&|M^(;HWousxpxO5UW38S(xd3@_@dSez;jCy|jUQ9WBM{V;oP_%;XY9|*efr5@=}(sj_C-Kd;lwdBUj2;K!3$P&SA7vCYacCgMa@RB@@;+vFlqZ9sMgA_z66+okzE-8KjT~*OLVo0T>++!)f5dmAbG%J&{H|s|Bm382oB05+YB8BDK5~QZYpy?H;8&b`$P49FAvpi#p-4l7pWnth7dUBuxD<*o{E@!f3$lUZNd?kcn(LNFx!bUy5MF~2kh8@j%nYMgJGw|kQ`BX_5%*zAF*qkT)o$sTiQ@m75F_QUzVzk+`4Yk@ZG#N}fxI7^1+OV8p7CgZXlR<3-EUktTc03SVAV?q;rkht{v6`qA8_*T+oIRQw8P_S%FSt$<T`?*yljyW7n7CVtQvsHUNTi>~&0)3?XNd|TQGzA|6XS!l(0qYx1=a-NOg7aS;h7Sj{MN1@%O?|-@PJ9g{RsJjwUo1Q`eaOZBQJ)fR!iAbZ(pY%XiMhX)O3=(Wp-3VDVEDxVtYOVWroxdXeZ&Q8KnJ;tz};(X=GoELj5(QL*_MsHO&lCiYN%6(uJX7+O6tUWK9jJDxd;!x;hziglqYuyfUAt6vIg{+J@3(7E+`z!#|SHvr(UJ6PHlVtM!T{+w_dMbgi-DX+y}C`5?8ci@MrWyQ^hv@U(zpLb|Rpo~b-lDh;dja8cO`DvnqU%KdY_ASc~`p2{1q_hJPNm|tMw@6-h5PQ6<JS8x+Gwn(kW4l8kDByUA61@o;L*-7GLylJfFvFm7K>=m(q>X`9M1k$7ez%dr@w-IxfFwqzhHfF`wU}3gdowv?O-aFjQiw=lJEj^+A5A(nE#`)O0dOq||>Dcy+nM>+(U(-@~Q#TpaKWS+feX2wW6O-S5$nC1rJt(RnMB|mzmZX?vdRv9%_LP?=V@(O8iOziXYIdb0@u!U4nk({9yCqXZ<TUow;LME@Q7%*0eMpBM9_wPtX;K$de>b?`0%EIDk5nLYkvodrv4eYyhn&2&a-@*Q@sTw;yK!zppzRnKGaewJyotQB;=s0VlB%rr_LOK-lEO7zJq6*YM5@%UOA2~Q4?m?y&uROrQ}X)z8n=?X_RZtN`@^@sd^7pL`wD*c{1r<VH#qcyep0*Z+D_~~?0!7^CXEk*wjG8H8zXpYKtKKc>Hh$n7D$l')).decode('utf-8'))
_ACTIONS_6C8S_3Q = json.loads(zlib.decompress(base64.b85decode('c-rk<U2j`ia{MoP)`Lk=UwPBm+}K#n$dK(2n}IMGAR7b-HV>1$1^eIQSR#3OPjz)wpF`SS_{juC-+R7KcXf63um5}Y@4x@?x4-^z_D{c_{q*VG{hQz3-+lP_>2ZDbbbj_9zyH_2{rBg;eE#^i-~af}zy8nX&%d6%efQ<B+J~P${pGj2U*7$EcYk(%_WEIScD`)B{_u9aen0uchxPi+=dU+!*LNS!&aY=*|Gd6`_~q<;vHSV^$A>qcUVq&GkE^Grzn@P#_Ws@LKYx0^f74>pw_ndT>kl8lwDp&V$B%EneA<0A`*1iAAJ+Hx`?p@q-@1L=<W-;{)7S1l&8Gr2VD`Fj_FxZpE%`Dhi-W$t{EEEm{r%nRbu^x+KimHR-ZpDDdF#u6nT}`EjxXQ+vR@1beSMj!;AiOwukYsX-!G55kL$<zBAS19xO(8yUCtNLhlfw|Mbs|NKmGsCIQVAPJ2sW=;2aL{Y?Su>dwut^G`Bx`-kFoGTXVS|uJ)zdQJDTJoi4Ed(By!f(5ztcmY1;yV>TI%X2#mz=ri^*?sVu5o;%-p`yp(nDOi^a;cx?+Av{|7*>cbYZDi4*lTY5ZrTSRP-{kWMhVbQt0dthin?8uUckDiVK6^i)58lA-$GzvlFTbRdKKA)^!iRKV`+p~I8v5My!&i9h>{hu5tjXjsH7<}bPo1Bw&h|Zd3+DC+`DtTDjA_B^hx_~W>yN+vY5n-|-Tk|NJv<Wz4PN;r#u6#N<4AL`y|pLp3HQ*>5t;osxXLe|3=8m^UjN4Y&ilBkd$+0m*J+ag^R6);Cq_6}xD`JG7$a~`;9k8fZOcsNeVFz(>ti~Az_B+BQs%0_PuT<6SfEeo1DQu4+K(OnXx!wY0~HUdWcw-`i2COF{1Z>7&-GP+r}S~qTQ-~rVBGH?*&2iS=5K)$Vq50zvmTe4ssuMXv0?q=Y2%+J-}}IZT44Zt(Pb1MAXzju*u~a&#W6G|xSdn$pl}UlhCnA&CtVCf3<QKThL=Y1Ze-y4{<!WN74S0W(bQM~Z;9qVy%97UqGX<s;o(+W{%8tN131kB013`TM`XwW4Oi*VlYfq-{o^2KKOXzzu_h)JTQ7F39t6{asC;5+T~^M_iZ5=0BSn|b07Lqshnd|~F;F~6$!R|ciT8S;>`o8H=I!0%zeFAD1&k)$(OrEp1dWDj*O%fDO~;~#AJ7g?8$jGK0lH8SKIprSJ-^j-W`I4i8<got<yZy)M-G<VevNJiWgo7T2YvrUbg4|=H@B~>=<t?cL2oYbhDvz2eQ?Wh`e7jacx*4ibLrSfmtXe$0gdlQ2Yo^#>cv$0@bU3(^V9nA@h^ZKD8-G~B^EXuynXT11BK)<ro)yD2`+8)BiT2i^!QmiZiZnvhp+k}B_oQ#f=-)b8BJ4rV+v7um=O<Vb*+!xhutNeKTd;Tw|DGh8)7c%z{q2lzcC+y;wp&xZGHXR%&LutK0P<I67g*BEy6z&sMF5lDmdTK*l|DNOkXQ%b!FQ$C$dyx^q}2oFR%K%5g#8=x`YX|EB-NczcYNLa<4D|VsZ;^9v<$$q^Uq7>gA6wGxYg@d?Q2x_rAClt}D}r&f%ms+%iT^7{nHt4{CG)$VTjTa^xYeL1)0w0a-tzZ~4(NFp~IExojm=s4IY~J05*Zqc-lF0#_24+I%X*kLyBM5i}8?nEOu^unvKK6aEpf7`wNEF#_$JqZ4g@HdOWk8*B8*92s!_w8-u}_FUsp0oP(?r5?)|x~k-$+CejjxWKU4DmSiRj;L*M9fwks*-ZQ8(olAGz3~nMl;P<{#%jFEc4j~bC^aX8uQPLv!vyLH0=)A)?f1N#5iO_dl3c`$o-)gM^(;k{2Ta#HPwSO21p$8DMWbMQT4O?m9j&{3AIG!UQ>Jy=zPCL##JT7TwOi&kZXLVlbz?ESZi9vxw=&p-?GZrkebA!W--;Qo%qB=blpI9R->_4~4wE^qbkjHL$aGH+J!Gm=j$MG>X0{u%F?X*?0ZWhW*ABt%Y@MepoDYgYf^P1`5#(@p?cHF#%a1Lla{#mA*stsf9#>x}rbWhnM?d-Q7JCtv)XbE}*WNhrXO>4%NQR)s&hC#rl)D-$ZntTtIBsOvZ+|O<Q8OVS(P*T6P;4J;--;<irRcZ^9fG5Ae;mJD%s;=o|MSbri`=31mwC=y59<B0rJvuoSmrx*$*=%>D#7BiVFB*M=rc$tw_wGAtu4sGSbn7$C=!x2k15~;j^;fbb828UFh|SU^jxsoSng(WdSv8G&+{a0R|pU`V|kGyqAlNIBKw7nR^(%~t;7drA5;$J9XJ_mcLC9|_5sDnXv<dxE{lka#C}Z9ef4nU)~aR>oa67+5DYM8U>*w}s`=W~lRQQ1*Z~;xyNh6i(jYhxXd{5TEIU(<KC$gd95hK9P_sL=En&=>g+5d<k~i=ESyn^?^d|yfR3M{lHG1wX!)g5g(aZ2R0G;k6&+aH;Mgy-!2z^I=72W;hcbXj3F+6x+tz-^7A>VpIP2Hc*!*q@Dqq+4x4u0b*z^or^lFF{G{KMW%M&{_ZYnb~v<N8w006k=tt?;q}Ru*uO)_F`kG5byK+KQkQsdGQQRS!p?Ofuw=J0<2uJ8RgREfo_BaZ~qNIwC6ht>Cy~?@HiL!5bgLDWLo(CcVYsk1k0T$@y-*X<)OGZ8-?$kkX~@1n(C7P1F<7>Dv)F&C-h`PZ$_|po{UZI#ZaHwa^-<O6amRUF!ruzR{1}VtWaQEQVe;Ay?860+x_#+81L?+f~uUJ3+I6DL3SmryQ-5wdxxktl_kPF_Jc;nFF;}$*F%bs8DvTbHdKo3)~YXO(k$&*G-dQiBbTcns}x>PO!4BpZYY4mF~!ceiZ5`SQ&l7kyUrHLIef8SBXOoW2NXB-6{i+$dUNfF)2UvV2WWJ@+vhIFU#^QCVRCWH5j1jOT@TuZ=RU?Xx0=MpjQf(O@ZNCmzn})lQfr@2A0W63^040&JJFj%`<Hx_|Iv3TM|$)WQLmo#l%7xt-Q52ZAj$uNxuac+nKN1_8Ty1rS@Niu;^l8m=<gmGsXm%7oH_3cgsNN%z%5N-*QlVan$-ccxci&$|x5b35P?t<=a>;?LszjaRJN4WxgI6GCUiIt7LLF><|G&X?aGkghNEblfi9b&LUnMWVxnOy6E(mfKP|om|9CwB};Y1=GvzqEsV{_Ml+zo9!L(=xr9lQyaBv#S`HyOQdo7&&=k%WTVKCC<gf^^qDFSWCkJ%vtKcA^HG`jJQU)b&FUtuQ9K88fk}aa@!j^Y_bkfE?Phdp6T*$?i$Q*`X3GhAzAcr9ja@zQGq$)y2GAE-ug~WUAfeuI$GS*8;E6dIfw1b6|cB2r<Zn-ABS=Qe5r22%L`sS$uB&E<FaED0<m$TKoxEhGdtmJiTe1eEfmY4_3s0QpMDXoC6_{JnM7mtoCG{*A5rEseCQ4ENKu{W0ucjd>gn>MkFEgMhele-5QNxfh0XdkeWx%93|3ENrKWknn`zMz%XF}fB}Mhp-C^J14aaqR*Mbpj|Z?sw^iOS(b@&9a3CT2IjsrPxL_afm{N;Kj;fYJcO%*^yNu9*bOB4*hlDQc-HDa-F;QYkHs1uZFE08uA02DiN5iDIe0IrNDiz84>(^$SpNo=R$&o1exi6D5panV<Z@}`*PZzn*_T#J~#tV8aMxYT~{gE-AgEA#^Kb+2#d*xNK#{Jlo=8dX=9W%#UVczs$aj^$sEx=7O*lZN-Y_JRnbvntP>_!VBmqWqVerhg(^ky7)GC;e<i5H8_5Pbsf+s?gG8)>9WAF8DU>a1(Y1ht=zK9(SDeIUh5JH92Bpl#%A6s8EUove<_uxix;Bk8e3{X&6cwAQ_sF8J%u8p!mdg@bK2=;XEl4cL1C1C)>TZA%fQjGh)E{EbQtHo1*sS8=2|40IDJBZSh~J4WU9b{0M2WV&?s<l*+}}d1g^ri65&w?T-tsh$TCzg1CZX7QfKM>^2r(k0_PKuabCUD*loCNb>lLD0eJ{@T$*2&mYYl15T0V1#5ER_*y&JguQ!&6U+g26aRmzUY#MA%`*GcyW^Sj%`nX_7zG$`>GI>DKCm7r1O-$}Hz=hJ5q5>;;uPP-<kl9OUm@4?L^up}GPmm-nBv?!1&c|n`)an{?vWpId*z*Eqjglm+sCG5PC9>qpYp`U+}rn)ev&Z_dLenXTMXhs_K7~-FGP9hgCWiF1y>a7r}5!OJ*B&8~J77A1Ll6bX*sY?CD4YtPm^!Y-hD`MX(da*<96XxHuSY%ANLioV!zxfD&7Us<h*y2SIb#Fu<eX~||PF$IkifU(PYXH&8Z&vFX_X?fR+<`2mV`PGZ@?pYnRmv=DjjTn*Zv)@`vi02;U(a{vSqo^V4)n|SOA~3gg{VSTO;g}!wP8e<yqd+DdwuI-{y;gzs?1Qg-o&OjVjg>exnt$=Q7V|}WkXV=Mh-sT)U*((bG4;eAEZ9MdLG)BZ!!~OQkl8akIhyeVtvPlg)aCm@#$GclEx_*sIo%BzS+F>scMLv`ofTlt~XWSuXt-<(BUnhaCTXT)tWGB4oT?rm~|oW0>d7iRFx-Bsv{P(QQUP1s>VXe3I(88xSV=kYq*oii|y#&g%R*K=6u{3P~3q~(L=|(XroOH)#+sL>>M`-MDchFYA`i!^<XaWMIgTaT|_J$H|V4ROSCHzw`ow?_7P&4Qkk>dY*$Id<U=;t!ewcce-U`aAn;V@X~X-hl2zYHF>Y7RVFAErnG9q6C)T>g`Aw?<B0-o}Mml9LAACCR*a*m@LKTlv@=3s=WC1kHz7t)%)Reh)IF|VdCHX#;((NKffAeJUq-mgyI)tfp>uZ+@+aqzcw6Y9h#duk&>{XxApl1ncz04bZpr!VLzf{QDUR*Zx3&jgiQ+m``+-$|e=2-c)g=ItHzzIQ8OJq$%3`F5X!ShoTCbK`{GoKtegIlh#ypzL+@&f8Ym5Y+|t)+D&Lc$${Vbqfgsl=5n(MU|2ROWC@KiVrm+?o<r#d!TRuv+bo7UySlSOU3_vwxU2?Xq8JLN21Dq}uR|m-<4ytfxObm%ZHo%L3w*4!?as#F02^$GYk~E^jE0ekuV7nt1!I#^P(V`7u%Xx|1*_9-sNqdcgcRL3+KcK^H63kI$#A7>xTU^V^C|4U%;1jO7FOKMlPYna*02UJrSr8=GFUg#bK@Y*%wrl*08=BwiJ4M77A|sE|T51}`{VIFhMx9N{@roRBFEbi<y4Y_js7l?B8qLlUPdDYoHACn+!h)r2JOPbI)FnXz_Eq@bLup<>YDpe*Q4Dyo)Du85scShyMr)}rK^rou2C1IEaf<oY#&e7eX$JNP851=-Lzx;|o@h)k&qyyc3Gg=wUet07u$$K+l|{@&{AId_2A{8cFifIKU649G<^f>J=^aJu|<Cpn_x9l~2n|B?sg3LH+^Va)z6IM^8V@6W>?FL9UIso(&X1Nmy?F3z^TeJc2_srei^Bw5GclAK$x3Km%MhH0o>WYkOEShn!$pu6H`AGS~To_G@J;2G1{@iaQARto9~i91{t5=P=Kp(X-XZnBpTE*7ak|AErG$3!&=OrVMZ1k4o%atw)U8fP>+&O~a68ks_m>!~bw-Ec@21{H?pVFYj(B@eV)aq^$@4qkW7SrXNiimhY^@R(V<s>a+URY9bftvIaKR7oLgjohK7YJ{th?yV6ANIhQ7{RD<(VphR(F_no+e3d$MS!bK4Wjq&Fk6@#8cC8*OWEk^4u6n0Q;)9UxMQn?_=~vd6ONlgA7ka2^9s7vL2>=y?R<5D_2GA$sTR`mF@nEmzDz0Km_Ukv4gR}g0B>G8h8c}mi!1RVB+*sg@5x}J-AB$NxvYNuV{@?r6=1-yRzftQH3|3OGQgI5Y_p-7a?N?_hfJ}#h`dcYLo76rXSRw#@iqNalPL>2Arzi4lsgwZoGVsV6lOW!<Rt}BGDiJGh4yEQRRS9rbR$?9J6EXd4`F5q+1-y=elpaLurK_;kwUI+3E{Fr=x~65qj<A=JT07lN&q6&$W)Q(t3+Y)9EI9A3w!n8bPFYdsFD6<Y3mZ{^L!n-!DSFV+uT^~-De6IUe`vl5v@L5H84CE*y5?GGScZBJR^ab(%G;BlroePmKN#LYhbKe3(j-%yc&74>(3ql7&@+Sed#Zs{`!uv>$+xhTSltU+zmoSA!NSktZ}g#+GrxUBGOacecJPLkTv6jIrv2=%m~W$62C|7bF<NC>T?_}qGRm5a6hQY=Ml1phK*<g_JYd6dmT5gD;>*%lR+g%1oqd*dI1Di7=hv;VFe}9}<+(yey0JvjM7~0N0K-~f0Gl8<!YYj?pquMm2%D3vVpfA*l&91nNVAh)_!`TlFpKOoEfjX;)H4li%KI8m-m=CNkl2wb_k-J4awo6))*&T*sV1P4t0(q?EI3*>Mcq#H88S$wb8XSdlrrWu_s~~aQVwY;<1IR_Bk+d~d-W9SNpRdK<V>}uiSOJ9=Pb{~BDyrRsxB>7TiC9^g`qhhTB2fJ7O4+Bw9@#BnU7>**gZFds+?pFuWjzo@us2!j#miT(!vwg={Vq)98YYeB~&qqjAV<<)Q%}oktVKDqtK~PEtvsIM|Ga1Rz*syg_Cx>y!0*x@xv}&MNr(56|4E)E7J^C#lc3OUalrJ_y%yqGJGJ#nhc-N&}L@v7o|}QDd*Z<O*dCf_o~&V?KE69of9=a#}ZiMhhBr3;FgjYxt*@@v9utVU5M7v5$~?if>fCaPbPBZG|ov)6b5j^I5?D~b!2T-)L|G_#1PFWpGjk<`he!8aZyP=iMm6n3$1%NR~6*k$pxg%OO##x))R1uCcwg0(8M`L&{18qS2c);sl`f525-v})V=KY#3<^{At0ViRtQgAJQbMDToWSN4ivRS$5<g93Rw5*0Z5q?!QmS2Yt^~Bv>>lc>s(X<W+a+o{?J@fHP2E@GE&i}=sqw6rcR9_5$uc2!j}E=97|g{QyPaQit@&DH-FYx9T-dxZ>a6uuz+q-wrFR*s1D{xifAqaO%Z_T$@hCD_C*v@V>HN+quVlBL#V4@RP^z)xKi`9P8ejN{J^G^a$!`v6~b1l<7s;xU`!+fM8t*0)df?MC}l-2qwYmdeK3>}->u{26gTTAiLK}&!NuaItt#-Do;s@KnlW|U);C!vvca+<Od$%nFcKBvy`#dKiQcZf?h4swECndqw?J}f&5JC9(Sg8);;P6hHY}-8u~g4xf`VQmxI$uElsIhgs7UHI%eAyw#JnR=z6>Y##>kr7c%o!kA!nk+Rhi98VKzxtB^O;|^b&o(D+h};n^q}2xezDVwvMjR5J$C@47)JOpz+C7bFDKd<5Z+!21GDPG`u0%>C7tJ6{pb<8J{LXViBpsWvCF4lK4-bh+iU0bzLeNC6oZkE5_IPs?lET(-<?!R#L5kAp(ro&SvvUJS!i<l}@@8t-7ImF@)+o3AA&s{#)AL8>7D$PW3}Yjcl7@D{oAD^Idwn`P#VCNQff-SD^ZiP#0@qynmdUY2#^P2l5d)Q7p``JULG+RY-SWv(r+>DmB@@ZC=<B<m$f$M=Y&Pj&2~TUni5mBvJ%rfvF0?3Cr6C5^5>8`pTzt6e_tiCWCkYajb$=6=`lJfo!des8CGS^7vMR$Yb=YbVyy=(&Z@1Sb&aTDq+Gobg<i1A8RaOxuo(_QlWT1oWZzbu40T|fX`P<zv^d>W$NuzUrHz17%87sK{v7DJ!zd)VPtR?s3vnpNoHaF7o`qeDRKR#rmMiA(U}O{oVDc8e}uw!$5k>?m`yvEo-~$42c#RUXG&GF8o+Y6pdht-+n;X-3K(&19r!+mn#O0M%t}qmj;_#9o$vGE_xRAOp0Cfe((&X<YZo~c{VS9bE_U%{C}ln#5GMo2Yn#j3^3tZcp#dc^&c0+iY*n!GaxClZQenL<z9)7Y=;cLM`}<jMoJw}&3YH0R$>qBpschxZC~wEBWBIihZ)3|?YC`LZ=gSF8DPX8qY`2wBZcBc!1erMuMW*U&Bp~Hf9kBvjvux$E51bCcM`w(~B09@(O$ONqI#uw%9&D_FWvL2J&aPucFGY5pN&@LoA&PhpOF-1g=a~>d!+F-JxK3e<F?1ed7b}^B)WNhrje{5FQ@t{fp@cGNn=VH6l<s1j3js1BJiAo663tqWqd2Ck*MLr0Bs9~x1hYyLzr%yK5Aj=;6!9}^YYh7*kC>^p7YT&vifnt7(2YcyuaI^?5c#3kj83nk8QEgRQU>jf>)=Y}-7PD;JGIUTRh6aQS#M$18e5V}(YMG~yuK&6o~?amt=cl6o0IbrUKfEm)oNNN7?C|Hm>P|IMQSj;!I!Id`?9_elwPYUH)5d)@RV*V;{cvKdNI;lVBLMQaOcVEj`{*58v$a&Q+@pvROWP?qqqFR&6Ae`%<7F+mF5x+E0bua)F~Bf11Kp{2KZK?K$1G7*=k86`dqZDv)ZU^xk_T$(&!OPruA)2m3vmug~p8yrxa<wYd59FuCx{&^DR>9ECcq$`~p2KkZ~UcounfyJF5Oj)yY*!hjg8p<v25twL-JYM^37%$thP{Jw5b%Q)}>|oH7PtnIdmNt@@%Cu4(KQv~8y?*Cj>yRAgl!Tq$)+Rt;^n6q^^Xt#NU^s39;gM95EQN)*?*pv1=pU1}k_O%)mS=1D~{xDnb}nu?%W0Vu>nEKp7mMQ7C$(wq~CZbntLd>wnE{fr{GR*J=L!M2=hSY{$Gy^|qVchw;B8!_%Wu3t7FqLstE4ulB4o2ry8rR_v@Dt*~A1w?LZm7b6fCLk)zXzDFHxhd3~R7<@T*j|UBjAIeL#Jssq`SYkvDtjxTk@Wg^Eh&rPA1ql=EnWaOU#i4n5wN^<08SWHUA+R%U-eP1KqJB5;K?Yb39DVXAd!?lvQ^ep1Hp_1U8BiV9EvVlRbG-SO+6z=vdD8R&l_2s@={B5krX-fQt8x^>YOJ@Moey{u&SsQsLtL^Wm$$<L~1Haaa)PnW7Sw~yG)tZbx`+7$WF2)TdZ@m6*YLC2()0^DFP=>`y-TjI5G2k^~jzdby+9bkaR*OyHC=wq)l7b<XLrXSqk5@dsV9|Lg+Z%b*i5ycp*=4CqBoHD>fJa-$97Ovok2e3QKb{*C|LCeM`jqC*#G^DSJzup2t~Y!-;j23mETkmWruRw5J`rj>s=dZ|7S%nTygGDX}m1Hy-&m7Rjrls2ei<OkdZ>j+d-P?Eb+J<aF0<msGSazq1t7mz&0HdPjZ;OV^yHF2FC7It+{av5G=mfo{-#Nb@@ZfsGYR(PPc*LsigCrmz**B}L9yi7eMgrmN8xbpBYTw5}F7$U$VFr2^)dhcR`*?O5oaVVR4bV<xTmQ=9^g+@^*GYj6t6lNI#-EFfPKd`S}qZw-!2nj@B`lPP&FtRL55<s|(OGng-`_HAYx4ph@uOC4xPza8UWl`?4<S8_sMAmVqzK-9t#Rcl;5=@G$f9joOjKzEpur#a9(d4=-vpihxD^ulDVGw4}Khg2a?>AZ1tpfZ6y(=lx|+$QLSUNpD7=;1VJl^vA(s#o#p!pZIG7ICvv8@-vOq+BO-A{B-p>Bx!Z2&b1)qFr}Zo1WgdvV`sV98mHmRm_O@?_hXQU&Z27aX5*LOo{5*V(Ew3d&9RmSpPuKjW6{`hmx*??(9b8467wKUPkaPxImT&ZoBvnv9*JBc7d1VX%ye2j2oTs2OFdyQmFtM&2&le6X>dn-VsA8E#)3d`HcwJx}{l}oV{5$vahTI-xSfo!VoDe-<BXXB?e8eK;4jPer<ZE8<^!8Q|z9|LoN#gSJ0CK9au5Z({?$NEkov}KjAAO1dH~O$PhN_@hwWwDK+e9*2}czGn#=nC(EZw!eCw=K*r`g;qQDOEuG?}T8fd({}ZFlrfsOzhM~z|(!u2cS=WZ;1he1?ZkP;eD`06gl(6H0-~vGcsW9EV5{uPDt=NFBc<1cV+{n02iGIO#n&^sAahXJ?UBSfNQk)8~L_{L}JZ=uNjW|nG5Q!2r8JHL!q=n`SWGk>H2xqdX&Ir$x_~f^C1zA3su!ILpBJM}X7p$e6jngM%x*K^hEVWw7rh5B21wmUf-=?OM<SnzKB1*Ab))L$E87MQPjzBvJN6jGZXKXF|GD#!*S`_N9AssTW39M;mkWxfJ0F^Eb71M52uOe$|NL2w9h||@{m?K=vALW(#M5P!`g3&gVCbN(tg&F>lgr1H1bep(@a$c=hEZL@K#HDME9ZwrVw#)~qU0u}GrrKRCV}qv!6cf^QmGMmFsZwcJrH6~kR#0)oYEbT<>jgRK2J}?kc)b@ZXu$je3xB63Fn8+R3b=xssIf(AMRr(;6C-&mYAKj+&B#s?C*w_HHIH3K8)L7C1ysk3Um}nu6#$O0c)yL9yM&3xh_Ep$z6J}k&FZ{$PV(O2ZeDaiG-~My?SGj6tvAlc-qrJ=e@e%;XUtqupZl7Y%A2~$sQyVyyXaFTN|>1Z_Cs!0o$f(V4Ivt@q_!l*EYsU6EVrk;JQ-_B7)^BMvsbe#C5b;}?ABb7huSTfA|j`;rv_(kl!$Vfy6!_d^zc|0Q%;k*p!&PP1s4!om3pKCnTy;}?2aAWTRi0CwUr}<JdTg7(b<i269R3=z?ks>3FS@Xl@$lJeUnsWt+%H{o01f+>FOy6M<r6FeqB<~Q+oI*MS4!#SDljA-`BX6<h5@eAKo9n_2rw%2i{lkv*)i^y12oi7xa_bUDtME_hI+r**9r?5VY+uWY`$NTLb#(?@#{+fyYZt')).decode('utf-8'))
_ACTIONS_6C12S_4Q_FIRST_YARN = json.loads(zlib.decompress(base64.b85decode('c-rk<%Wfn|a{L#bd0;(QBz5C-*J>Ke88%3^3ade3Fo0GNAgm4}-Gu#j^^*0-%CImu^N3`#N4yn^#msnzyScgfFaLY?@4x;2x4-^=_D{c@{qW_}-N#=)-#$Kld03xq&(HqjxBvRL|Ni=yuOI*R+wcGR*Z=wY`IoaFKRy3d`|!h;zx;ap^QWI~@6OK8KHP84&gaF~k3X*0p9g<<T(3WV{d)7``u6GU{A%>|PwTt;pU=)`ho66bxc~U&!_)CUR@?30&xalR{OQA=zkEKvX*THFFK3(e<I{6nf4+Zs`tkYG;j7Vy(}8$g-`ySGx){H8|G2@cKtqPFJ$@Qb1!}<Pb=BE}Jv_AJc}`|0eck<vyzBGb?T2-3JW+r4{{Y@LYBzc7?q7!ES+wK%yPuDX;iRv-nX3FO9O3ot`2EM_ar?A>7%!sncc-fdF5UTf5k202884!8asKHaJLBY=QSaDPmV<LTz@t$*_V2^(ZfWj+^s+MtUAN})I9%mR_oFcURXAN>|DnkNJE2&?<So0g2V*uEj$+2j-{>>88+ST%C(j-4yyFm-(^OfPGvROpo1uEN^0Vcn3);w{LnofReM|MRl)s7R5e(t(gaLCD&6_@mhj$!4d_8*~(Fbqfj^p0);N36jr1yP3o$xLl*#Ga~O<kWGe)tBD9o;I6iZvM=rp5)*=c(hf)!DwU-h#0`LVjA95q(<l;r{M!{o(1?Kdm30KHYu#*V8kh)8M6FVl0vLJ0_Zg{jEJ{PjwF+9FftFD_8mD*02EI^!hjEcihKi-n$L$zebw`n0JNwI55J&!p-;@z!-sh0{3dSv@J84_hH!EsE^?Q0>|DkNSUhwKSd8@V}U+}4`d#JXg@aiqxB{y9jN-CO17`Efv9gD&p+{W+FV}+cnTi}y=B9B0LK0Ck)<&hZ~hWEA+}}QKI?IzsY-COS2nEQpVt3r^1Tmis3iulXH7-{0+K~jgI#QGR~$ogDz|fJ9VD*7$Pj3R>ZFUIi-CY}#_FY!yc-#~emt(*Mg_dgc{DW^z*}nbAKnO>4UsZW$nbEhEq*iwr~#a20e}SOq9fAffQGBI>&ZXH(*8Ke+52PPA8TS#b?e2B)q`MqB`P0ST9=tKGvkY!;7HQtGr*9x=wW1cWegM#QgYf)LgKAnD7(`uWAo$f!@txz)(RL6x}&@LVh9=y)uAuRAsUWF3qPP7oHBrTU;;FuAbijd9ea7J>C6Co<S;11k&3Yl0FIn2yW<+&56UrIDG&PbiRe-pzHe+_S<&G=!Ghjg;0>AZaQWbp<MeJIyg#-T;kk5dq|4_$e?sHC)<K`D5w$aw9-kg=H$SW&9{vJ=bSZAcF0pFEmA6|^NE~B2ZAq8l(ndd$eIrVbpM~RQ7=~l`svS}?q8Kdbw3*6i8rmB}h|0r^crdGJeH=a<F6sDj8VtL?V+Y$1b5T1+9-I7)@dzYWL9O4`*DuY?+IZ;GOG7ge&+^_P{4;?%?L01n^G(K%dyg}HtEkn=wrLtHmBlu3d{WHd<W*lc;=}!uXOf`vRq>Ch`yJscnR|r+5QAHAbANyLoTdVesNElTGxYU<{3t{MkG{ASu1nL0&f%mMnKz7_(1|TF9@N?eARDp!$&rUV2b}>!2W0*9zU6zzz)0dt=CYMgp{4+;?s&8@jmo%h2wX{EYVj!zKduvDMbJcmVje$bz&Zr_4fsdEVr<?D#t4*ij!v}k*^t=_Y^>2Ib7a8zQzE<L*mI6Y1zd}fl{%F%w94e5+Cej5MJ;jIY#AF@Fh|t3xTc{LMK;rMxfGP0LvOsp0A+Z(k+B@FvYZ(Z0t(H^%GZ&(#$f{W1OeW8oc4R3&WM)NbxAH_L{Ayzyn2=*$^)kBou~E6n8GZcm_sxQwx={ERM^qF%a3t9i#=spr{#OgV?&&awotoeZsXRmdtNsdtJiI%A;zr?Heq=Lkb57rX!N&ahAXoP!Vf725%jm(sbq)A9A~=e8+By5hlifj7;afA!y%cN^x=tyS)+3}cInQG7p*r4N96=10*c3{7Cra2h8%&#>C2g5iS<s<icmkGA<mr4Z&5QS>n=(=Pw$2}BQr;uX2|<1Zz%XP%L7RyPf+9N@W&R4UXDd~*vCWMx8?}!e=CGtEi;?bM6ubhe>i3mrGn-jq6iMf!#wWq>z_W|{du>*YF<+L%Q%&;1q|P9-<S8z=JBpvkX?C&A%Y_Q5=}5zSvH2iJsy2t2|pLCL$I#}?HF%&cu^%GvvW$dCvYwwxtl|)r@%x?il^tV<;L<blf#1~riGpdIle+P5$v?k%`C|o1C6l23Z<>Y$7ma_4(1&gq-=Kqg|qSj$uMe3ZUrud2&2?pf|w}lfzGAP%)C9vIn2QwV9dZgCO%}7yBnOcE{!AGpiTu(;ucDK;4q+VQ2LP*jG#BG<uP1ok~E$%V$hhdFSR0aioru%h(KYkN6j!4zbsx5a@#C_UHM={=6ZYD-;6W0Qh4CG5TWmRFKhQX`JJW*HHAwKERe_}C+u4<s;N8Zah$F&XsG?{LJ6t^026<(p((qyR<2)5)+BT4%T>(17d^(S6MA%A;};It60aq26#`F(NDj3{j|pjQS;Zj)qSf%mJRO8IyHJT;VJ+Sgx1ilN+~|^xiM6;zr&;+2kDpc^S>oGxh6pf?8h{Fo1x1k?$*s*~JDa{V!$*<q3E0PEB_-Wz0K6CWSw#wF3~Uesy8&G(7cO&QB`cNLt1RM;K-8H*-(|w%MtrEzex-0#&ZJK^Od3y?dY)-OFE>DmH@-%*@+8l0j$F$y0#j~k-!8XF-a@P1=wLO5`@i7^eaq-U+J}y*Io~eiPKYry_sVkAMB1JVL0Hjuxs}Wa%*3Uxwr%pnN!JTlW0O!u!8&OZjP$(8HUcQ%yRwB4ME=l2I!agLS24J%nuF9_N2kPVGzu$CnJdAqevUKNMH1t-QzV5+p<LYR+QJMm;G3<!or6qPw#HB-n5?!6Ag|B27H@_f0u2P)ENt!U!of$#!NQgW$UZrK1;hBbt=DC&qh2iHYHLt>DMrFqFzc=w;y^uH8Mp<R7dCldUN~BjZ8}LBLE?04A(nd}&<$%U+RN+Yzq7p~*(kh~G)LBtK$XGlxKD9O%9+q#c8&qBKpPa4LQA@snaay;CUcHrtclb^I39qyEq8U*n6y=biO5!iU~;wOV$1_svt3lBgP~PoC&t0SuG}h+`6@{7089+E7%m2d{s&7=uH&PjH~|9qv=?<kL(i)-6OF&isn`$0#qe-ElCx>m-xr~iPU+Y4MK<MugEBf6b2=jPspuDR$zaR%o}7qrWcgEy`V{y_OXf1DdTYU9ED|c`f)7V3qfR8VESeKQyyuqo=&+}HeQ0IbsY2uovIx782vO4o#wiGtMH~wf2K>m%6@I<VphO|E=O&+q<nks82Jo*KKU<}{tQ({zOUh9xjYA3l64sDhD=}7pspiF=f87NkS!|5lR#!d>)=6Dd-aAiNj&UgdZmV-roF^i{H|sYIocnSM_zFq1JUj7~IxtY=qygqW(}c%z8;P#K%z$Tt(O(OIS$zg3x~()Av!$gP6EwV?v}32K4UtBYbW>54gD6?UTrKRzOAEzmWi)AP^d(`OP5rK`k%U6L$rufe-(l6utRdGb3x3ojBq~HO>3|6a-%KLv^g$4dw(*>2?B=qHkWPT<^n`6(fQKdfPX*urBqs<}7Fg79POh7!T@S~$o;;H$;4rW|ZamDEsY%5NNDFn_qE7m!M4fbX%%D0<Mpwk8^huiNGZeBgWLcv5=~Me#Pfyc$D<~|WS)m%8ie#pNQd|j=S4BqNNLMI#81=?ELnMLNl&FY^1~3W0lCh9OYDgqrVw=?hiQyE^T_qU}B=YjMy7<mgN)LnD6q1KXO>$ggUhEJ36`^5@nom-Nk2z4>|Dj^I3HlzGP|eR{@*a|8d=kBp_-9gFSrhimx_#iVRp+=IP?fEj_ECvFs6_24Ju9A-4z)Hpslr{X+A>%=^h0;clR>VdA^s^UEd~ah41T2mMPznto&>AYMInBwGN$s`qUJ7|X&o<V*4Gsn;Ibfl%)Xg+Mh`6D)zmCYt69YW1`8Y62JtF!X&a@b-%JJo)?)!=h<X5-A~G<0D3o+#>;W^^!^4Ey%Q`7DkwFm%Xtr!xVP;_hX>$lVIq|*3R+A(RNkemu<}+HEK_o-WTMC&kjnte7m(jHPz~!GTa#m(_kOEERI+k)91QxrW)+4PB>d$i{cJHH(QedKZkBCtyB2Fylf`Uu+TosD_8}uz8gGC3ORQB@H^jM=nReC<fY6U$nHI7!aS_=pbn@4iU<!Y|k)28A%#)uI;Wj=w}B^`cCae+)#11))txRaW{$D~)aBZg>S@mw}ZBep)SwqHrn9I;jnt-U57Dj7IlId7vCDAeJw$CH5cd_N0&rJji*N*U7~Rjo8_3;#4jj7juhRm&**9w33x)47=n5eWQH!)LI3FCoy6Zy3`F_FQ!t>@brl5=EuTDHAj2I%-4s^ViSQZ}XDH5t1Y8H;px9!l?P-4_(wchv(lilRPE5aV-kn<)q2}d8$yIG-F@^4T^~jr^qQtD7l=fx@LNRYyKNiUeUNIhD%1rox43LUy!CiXp*LhlN@y<a&;*UIJ4kt7pscgHB!noF$HcJrxBm@N{|Q&-aufL5;sa0znax+@S@seHTYn+MuH!nM#D4Lo{ub<y0Swg@e7?d)7jBkmdY8+KZNhInw%P~#z)byBuk2?Q`#l-Og1DWy%U!ct;Ro<&w&bFDj=O;M(Y|sBCA;Au``Do;~VpBf=G=j<M|Q;VISm02D@fSTLNRUM9BvyXlU27>zV6SMd!3mK4m*zUv@02oi4_~=-Sl1Vd89IEQm!SQXyBu^)3nhl9#T@V=4ffxZfEWuT1m+3tW$fOHOJ+&X8rH5*{qq*bwt8W%)m))nMg|#UZr1;QnCWY;ztey1u2Ncf@zIge0#86G@l*;6?>Sxd@XbUNo0h%uS_FNb0C+K@L%dPl+B1us(0H1`nV<S_-9|G8mYWrGY8%IBr%Y5%{$n<ibr?KQ$atc(t8Rl8`{y+)A>)DsI>^lR~9o%1#%LFfn3YeO1qCT_l+a8W&3}w?yk|wf1msZBJI3g{ljb5)4ILzGZ2)D>&BQ@a#0rhA381%>Fw>x2SgG3;XYCa#$;bT1fIgq&80wI_{qOpCP~5<NzbYYn4RL5(4_B@xKb^Lv}+AErB#$U<3v+IpUXo^&`>8Fb=r(Xtmi!zPwtERt(o?Xcpj*CcJ@K+(;l%vY`r++JX89@}0TQpXJ36J#JJcF;+vVSV`ex2Qvi_0g8n<c_PL7&W8F}`frImu~cj)Ko5f`c(rgIgrY6$GcZRZ5s)`xZo%nxWDtxO(q>e83ZR@e$gRz=5PegyC*o17G>tnfDy}nEtY9Y%KF~3u`2v3YM}#!2;Z*7L$bVU=z8;}Hw0r_Xoj@d|9FPdt3Nk+okJ>>30&sAxD{l6|-1@o0L*I8<xXFO)as8chJ-T$m@`_<CG8U)&7F3Un^qERuitQB7L*A@yyaA=J!N5#LKaRP~9pDJqv-X7fZtiwry@qb%DNSNm@(CsA5-5Z)BC4g3uAKx}>JprW?S*K)4xOZmHL|J&glM4iwWwBln%tpOLu5x(1PO4_nwDCafF5%jiFN6uxu!~4G|U_1VZpOTt4G<$gVU>>ia~Uqev@Mnm&c$~UAKbvaZa)xXkKg$R<=r2P>G&ieKAT<W7v+K5eAofjo{7QU405xj+7RtIKY4fn=~Sh|BW;Lt#Zgw(4PVjwtH7}4J#~51gP?FHPly%dB^VAQiOe&@e;3hz6YcM$wL8U(;b}$S}2P0+Y-bHfkxXZ_K14Me#ItdF!5&X*XJNH2!GEkbS&N*C0qLC)eD6$R28^TXqD%Z<|Ioi1F8jDm6f9-N~%A+d=>YkUCLmH`pjy2h$q!At9a1tyJbhbGWx?bP2L9dQv)3$BY-OgB%@Sns!twQiMqw$x<+S5B*!$D3he_2DTzJ+@d~5PYUxV<*1mQ!dO$%1pnHxsMuz3mS$5&7tri1L&BswxNmXAhJ3lYQc1hLwi&PN@Bh$=Dos;*eI2PC2L|q7Ut<Y#aPk+z&yk3S-*@aO1;v^ljWKBdsfFHJ=&9@m`Rg|El5Gs#p^m7{7#$mOlrN7yR0Pc$tkEh;{CM_sdZcR?el~E=MA@q<=pchpk36Oml*#sN~VK7lYPs}&4xd<z9cR)*2L->il?7(-@w116dc8Pto+BJIKO3b3^cP#CcK_WcQC(oO>c7So9O;q;h<EMXS5~(p{WA!E{t%-dcrkBeAEUFo6ck3#jZlxMEJgQ>Jw)#oa*ZKZwt+lc9hKeDb*X2gbSM<T^3V!->rw@>>>usJJrS>~1wo`rYY*s<%Zr*XX&Lv1F9`p(tI}vOU@wQRI9;!%kIZe$GOc!-J<-sX>x`(31kr3t)njX~7$dq_!tUW+VUAC$0<o~YTJ(t^F*;kqs4v7S|?{S@ybk!X}BL1Q#0Ew})kyu(wWfVhzhi@UmSL!O@$RvjtPfGhLH7N^YVNs|`%N%3>sUkbeOf|{~g#}(VHxg_3wx+sL20~BQ)OkY{9>V>>!6-!*L)YW-&oZS8l&ztt$nZ`j!sSevC{ZvyE-VWs!p~R{O1XM^ZfXe*Y~;zqzi{NNj#==OK^CV1Dl{?=X<IlhE;A=Y@g;0e_khLJSON0|<Q+p~_&N$L#o{(pE!r!>O84`0({>r3WBg#<u*5m9{#cv1%XzX?wR+5w=tvs4Z3+D_rY4E)(5^XE1Sb_3nuvUMRRlIx-)rXK>i}MIrPsQqz;qNdFU!r!H`FpdnqaP22c=cYI91T*6`-$RQR$Mv)pgj_X_Qx<nX^}cV>-;5f=jridl^V9gUIdaQy$P2x%m|!O!5)+PFyy@S{yN~QsW7J)CU=|fGvQhR5RH`ePLBUf@3R-#`?fw5db<duC&O^SWn?;!X<iQSgW3L>i2^nKUmjEUpF*B5}65IdqQ(`H7Hehgnt7v083Ful96-;Pw8=i1ICRv;Fi9*pUqjW5w2Sv54V<dCzY|)E;6M60nM0`70p|8%&*ee04ECvzzBf?SQRU`U<XZ6HP0bh;z27i=a(i}5Y~R6v>Zy&@v5+g)|U)C9t25HcP;yyAco3*rlhlS%HSp6Mc_=<&bLZi>&k_dgIHT!Z5Y%|0X5Z7sZ|9gOCvmBX?@SA)CO4|RQ-yU7!#VEaMuQCP7!Rzjl|!I;777lDr1U4TTReI6qpiC*<=ph*DhdXf`+*SM=g0y7v*kaTlKKD*}R3cBX;7m_1T5o1rqy+HgiHK-Qg<=JvwgyWpzD!BEJ^OQ?qj~6*N+Uc!Zc&A+QhdsB6hE>Nks^cEq)lBxs1|>#}Ba8>Xgm?&JwV|BrL7<jddpNm8C}MnaBa&h^aIHXehpp(=t`#RAmB0W4_PkVCkLS)`@C0YVcXE*DBy;LE8#Yra}jdCbqVxLss<;{S?4Cp$|YOyZ4`^sU11GNQzeo>X^MDPTJy$H6TLR&toVn<NV9mIoDcBCBg1A(Ua+^1fy1ZhF*lizZl$*+t2(MPAx!8)6i-BELB6^*mdpf=8G%opVjY42{E0;4c)v?O0JV6kNuRv{o0hK3)p#H=j|eL94WfCEQw*7Bt+3uIOD!6szh#&#vE-COZ{ytDD8IO(M%fLu&Dx&AKa_%uv|R=}&R^Wr|eDY-S*x038bJlKx2f`D<F_=X97l2bh@K6OnE3w*Uc*CD8j+nTtFdmkTuwjgZoag8a~OS=31iJ=EO@7OD#ZQaWLknBzBJT13aqs-GHo3Zi2Pp@)8_$U>q}7rRR}he=&VRv~h*g6n>uk@sH5TE=T2A@3N}PJ|~E(?G0Y=!{{&4%1p%by61_9z9}WALSPqVC99A<pLr8fHoy1EhqesD|hv9Y7yvA4T+-(J$p|ph2O7)9ln1-#Z(|ZZO!E?c~f8az{CAhx+WWjjC0;E!PdC2P%L3Cn^<k|-LB8OF*QY`SSW2MxK22q0k&D?Kqa->8YKEO%qgC1dhNCqnQQ?qOaHRm%WMc@K&yTnWi`S8RI|vB02gKq<5_f9L@6<>hO;*SrRec8KO{+|v1<c3Jz>V3fNtM;5y%oioD^S+NYoo4S^hOd$-??i2JjS?xnl$&F`la7ZeH#wC(K!*U{pq?5}Ht!(`GUboiKsn2v}5qJGRUhE%U8qFn~~<HPTn*JTDjYG6w`I$_#PBMw6<UcgJT?pxUy;ztD?ns)Q^r86uwRi#GgG%TH7ewKf>FYQ~D@P6E%2FymCoh86%})wanMowifb<0@PMMD(3jRL}4qS9}HR@`5fhT5kc-Cn%uJ(h$_iy|LzY-X@5~xdPEz4(m3xxu%_Z7NjE3Z1dW-HP7Z+=`u30mnqFSy4B9Z6lp=xWh%-Z^I6y-Q+ACCWU32t(cO$H?M4w1XRDFG@O>tV#cANnF}*BgIM8%oo02%QW=|-TVrGT&Av&yZED_&+OV)UM`fhZYuj(@0tQ*$rtnFr`t_?|$l>~gM%bu!UrloG`YW*qihv&c0pb-yEF><a?Mz}F=&}=5ZUXMJe!YZkEv`)E>;LfK;aBDkM$2P08IeSZ7NDv}?6B2to0KYcS=xPre@i7wu6yYVK62p9(t^FD~Lut%22z&}Fu*yZ>CQ8E^S%5OTQm%Td2{PChT)8NtzD4t@uaGvls}av-mmB4V3Ca9%;fKzbnl23xQMQ3|rPsOOt9^3VmL<k=J&Bk#LN(B=5^MGrC0+x;7Gs4>8^7>ynw67{$S-uE2rZhQ*5!CzK&Ll3V1U<iF9{<w-B&zzy*kf*yhc{8mQ!CSyQliqYkPxYO%Ixkr$;@F@)I7YllA@>dtIAA319C6>Fi=1soT<9$6~d|@az(})xDutk(Z3CG1Lp_R@36mD>94@_|((V5DFX)LpwDmE#cj&i;PLnQo^+oQ*nZPGB#`22!5qetqi)l{Z{6EnXblK&Jqisf}N<&Wd>_rY8y0Mr30^_OrG<~en_DPL7Ll{-dP6>HN%YI1*_6EZ5FRic0D=i6~Dd%RC@6Sq0(38yEYWzb)@gpkf<TbWysYW?krSv_i93hYg`S@p~^K^;Y}=<?y3EeShX!7-@;JsB>s3VDSEKhuX@F6sX-Soek_i$US+ov?R29R*(CKNnky76+aME90ykB{rZ%1Cs*G{J1Y5L4QzGQ{sJ*%BO*R7s5+ox%VuRCNo2bzur9LIj(qx>^sC!gpeoc*-D8x!y8_*$tr{tPqbh8BqV-y|vQt?@F^T|RfffuP13}CKln*`#j4G1!sqgtVssIFfMvVB>SSI%*i#!?u$bR)`=h9N_F6dFS=Y(WRf!(2&`^R`|~l9XC=1@c?~|FZ6RALLReWJ$`o`zRo-i!0i;KDOzxWHn%jun-v1#{&=q@bkH%^B#z<BHqcw@a6ND(b{^*JgD;uKr>e9Qfz*)r)L*f<Z>c5$zm$F^Q3<0r5xVceYIB@`F~K$rV{{iOd=RHaZ&veEKT^kYGf(J9bHzZi=3RL+XVxeFslL^b7}ODSzGY^5ydbnqak39GKJD${<E}KCCjrZEx%S)%d_^K>Obg;5>p_p9!PAxa_z)|r$CF-A`lI?XF3SKVTztI#u``mwOM?<k`Tdx8%_<geiCH8UGuM+OmPwuU@jvo=`Bk31r+s#tA&>+rDEmhv&gezf5y`#RLAzN4zM(VlPOhOmCeqs9utJe%LA?in)gLq<Z8%0!#ThM;zX42<e==HI?-_IfKC|$c1tpBTuHQ^T=S^bXz?MG<n!W?T^5^2pu!EWN=q`Q{P<6M@AT!I>7uJNZaZ9CY6<yT;d!O}?aMLKc?VtOF_euYNlx}vC-U(kk57{zDNi&$oyXd&Dry$F`83w3R)1w3repfO2vh!=N-ssBS#7e2hG$CgoO1tdYa@`&-w2|JXf<}3zFsNXvz1;z`j4cgSq9*$49xkPVIc-80_z&w7@BDWY2d81sJvW9E?z(<jP>|Jju($ni)STF6K7x)Y?i}i0%xWP_ep``Bn1+t81Up$;aFT|@<d5ACeC=x@OrEs5R^#^d74%q@tmq8naeh?qF73LF>Jlkoi7`ZA<U4hB3g?n%O2^H(qCRU*#gh;$s5k7)LR22mx`7u1SUm2t+my*#G+r-X`?}{f}h6rfiId!hBvswMV4oqdRZA^2a`Y+4Y8N3MMJdZ;42hOErBbfUD8$+ST})6L43s2a|Q{Q<O(ZaOzo|laXuHkZk2jB?88A%fhXG#L4vUV4LQjYWGv-BDjPs3R+cz0y()>B!XIV%qDVI)-I+ydKkgS9xYv|RxO~v*1A9ygjlu*4psSp`61UnoB8RGEmvUOL{JU!W4xjkrNk9uRJu%B($W;~DoY(RI0O&LC4Re+A@CGX{*Y2j}BsG~?YBuvn6=X$AnbqrywrjJ&fn7A~C1+bsC|S*<Fo0%B*1gvQmgdxyQAlg%vgn{LBWWGeh+q@QRDf1eP3YJ&4G>B_odS$0v?z4}7q-e+mDZq=q_v*-1xiHbH4?iwS5eYn5Vzv>h*GO(BUZu)REE&tO4y6yU&O4Yf84sakB?e?h;dV`=E;*fFh}<-qfI>g(Nu_)s0UODt7zWQLo5Dt_#p5aOQ7K8pV8E^(Mo)1Y!)nV`UcvC4~n<YvMz0SZ`FIN-e9Z7aWVGKm!-#A<7<@FudbW?*g)<)P%Cv(pA6RTMvmpzYmj*cO(>=m41{HF!ZCM67^@*-@?yZmW5y8`LZreCu@@Z6LPcyWW{?SW$%%Owfw<OW6U#J{fN#t;c{As$fwnMtUI!JAhL$dQ#jj+6hK~>TpX*jLg-U4@yC{NUC^d*CnWz?H4a@)*7p}mhpU(g=;2#u2QHf#|_i@-;gk6BQglhBNy~3(IF_e^&M-8C`PCotyeMS4aS7#+rqRIM7&(B<EcL!D=p|L;J2IPkQSG5n~IaZpY`;+Y=Ho`%>4*RV<i8qx)keSV>soTGA{||QnE?W')).decode('utf-8'))
_ACTIONS_6C12S_4Q_SECOND_YARN = json.loads(zlib.decompress(base64.b85decode('c-rk<U2j`ia{MoP)`Lk=5|uZN&D}9pGcsg*h0Q=143G^11e=FR-h%x1IFd+S-cwy&)#p(5IC{ILDc<vax~r?JfBEl|fBo(EfBgOTlYjc<<cH7iZ{Gd-;ripJ&v%=XhtrdP`|Use<v+jt&zHx4{Pz35|NXzdJpXd?<NL?|)gFHM{I_4Pe}4bd_07rY$=loelhbAM@y8!Gn-7!!__*1;`||PqkDKdHC#RRQkAK?S-2QxWy4ZdF!`<z>&u>5N|Kj4|;eSr29sBV9?O#5B*uQBp>Dw<S_nVKO9^3l!?cJvzAD?y~%^nU1;^XG#X8+c+`CGR?H+dCk$n>@Qr}<Q%2FzX;&K~UHt|gCivN-7L^S8*mKHOZt-9+Pw`m_B5@U~gI$y=ZQWICQrJ03s#dA}GA`uaRm!Pn9e-dxY$zh55LpEh^%MKu5HaP`2YyPPkgkGG%ai>O_kfBL_jaq!8kcWf%#!8sh@*(mM%_xAdEX>Pytv@<7Nx8`y`T<uG@qcHteI$dD@p~(R|p;^J?Eze^Q#%wYi&5X6*(P!*=-09FA{O)|`?T4_PreIwzgu@MNhVW?RXUjnsw2?)JPCj|tmg-|Ef0EB57{cch2Fy`5Z~7qa-m!c5a`t{i58lA-$Gzu=pT9{beeCbk2_Mpd?cYw`H1v1Vhp+Invs>jXuqKnk)VM&#{ObH{b++$|w_t9Mkgqmo#F!Smy}h~Fy#4g+pEh@&-rv0c=fg8$(BPF{Vl0vJJB~C5+gp3mo^TKC9Ff_VgRA`f!LR_o>Gf~S@4Szzx_6t}f1Nf7Fz*`kabkpng<J76fH4C11n$-I(zeWG-iK*#vp%K+2poIEAZ4x!e9C^1jRks2e~@_uqW#$6kH$?dI#BVTO17`Efv9hu&p+{W`dnWHcuIc`ddr6M0F3+nPqxNjzWH0=gxHpO`>dZ!O;v)My|7{Z`fKBVO}_Vm4Yg81?z&+R+Y0UBd<dg2X0Z5|Q}6B;AvMx*$gW!HkgV7bySGjbEdTBl+uqYTYX}jt-gPI?`?bs1pcidrShyV%LXnQsl(pY5o2cbMOooCzMi>1a^-Hl)f?g$qkwb>g!8?btz8~P~^=Dsy_7C{8I)F98)QKbSFod5%PUkj&5`^U2cQ+m^bLTXCrRX&pcuHRYGP8&(Ac%)bIqfG=^<GDoUGTx!{CIu$*QjITZhQkR5Tn>=sCIoR4$*WhdMF0%;IuKw9hsmDNa2IN>)6v<y+KDt)oxIxBbCD;0AD#+cKbED9h7~<Qy%pFFQThv`o4*Qu46EBjt0HYz#A$N=JtnMn$)Y=@cOg8AkledIX!=G{kYv*W9l3e9~X{k^=!m^{B(DH|HJ0)?r*@7DIrX0hr+i*8s>61+{79fG-B~^1T^XeK`86)G|b336ji-TV`QNUJRQr#np!7QtjR-|IH=OKK6VegD?R@?4QJcl$dgTr$*%)Lon79^d<2TCAnLdE@l!La79o0iYG@_I+TL=6@FsAzou8|~gh!*}y#}_vR+#L<!H&*5?V7WP!tp7@%Mu%b7%J#^Rh((+t7l+L<yv8A#pDvazrDSAOpAf0)$@NoPtce1@!d(;*4z8@xVOf~($T4zgN!0Eh_f;u>gZOG4c^09vDfl$B1BLQ#*!}u_7BKZ8f_?)ril3=T6|2suO%3&iyo%?E`98%Hu{+&WfFSZJeBdzO_U$uH4(t3^Kh)!M1(R<I8%p-1>L*A=zKff=;)iD7L{$lh8aESg91*S7C@e-&eRxRz$clRx5u)gF3fD%MJZwj%&!HG*{yP`3}&<1rkY5n(wty9^flvLcwh<2>;;*qK^wr;%&s~e$xwXlY=DA~+gm>PB8MB)vxH4H%;wEgw>|4Yy3=X4UQ7d1;06Ys?V?Tw5(wu1I?1+Xr1D*0iex({xu-pr1=~S1`Bcv}XZfx63WR^qX&lc6G9`e4G1G5g;$O6ZVK5bO8;f9M7%s*=W%f<Q*Z|f+6QXUh;jIyGJ0Ds)r{e#W$d>FK?L%qA+r>@BlDwQH3Io))JD(=ELEHbc&O0`&v}^27ijGnX@&ed5fZW;t*U~&dItsvz+ZOiamdNShUmo7Q|Fem~0{dD|9IwH?fFq*$Dg8{r#T)E)G+^-7hWP3I&0h{(D(G+_D*=9ug~NC6I<o6^mP1Lq>lImbx+Umx^a3Xgl8S&a7&#Zst*tOTIUZGqCv<4O;%Vx?o}SBW3xM?zyf0fUt+jgK)|1Q#Wh)5;jq3(K23kYN;gFqMMRQ7aQ7PeCT|AmsSpvW{Mn~(|?+top^I!sDA2Qjx62RalVJ@x$7CS{|TH7^=a{zCMIYF5AY1&D|RcHqiwVt@nFizG(4pq?S(<D~Y=ni;u3Yeu}Y64@H3@zFQU^8`F!6;_I_2)nnrz@|e8CUFN&{auls+H?&K575Fl2)+StBc<qGL^IQkhKiixr++xhrKr^TOa(Alsh9dayRQ<LnM!;kPZpvc!(1<K7(?|!)s&y@Qvw?86GmE&#Ag@S<HZG+3uwkum~Xeo+e{8P=cYg&vb8^aCp5CXHM}R;!%^}pG694e4#T57QnWKnps(tg&Lg12>|ikDnS7i<T+Qcne=buHwm!96V+^mS0J*mD<BAe;{1~XO5DoG=WTkJxMPVj292*3T5hEYIQSUNh-k?golZJ;#QEqF|B?%1s7KR%3|E2$5NGHH(e#Kh*GHuy5iu5vWD!`}vp(1+p=L~CHDcUc^SE?g7V%a2szbDn*w#a9xzIL{^t*inakeC_7N*)#$I+R?NBps|b~I^LZPbRe3YpEv85-WK`oVK-*GSSfxweJ*zV3?qimd_`>$lLX*M&7;cj{QdG5!4RAp0@g`AT$L3_|PCapK4y!0o;(uX*sYmAo#cumr&ai$`4Xbt?=qJuI%f=TVmgp++ykp9R7t=MRS!%Nhh!gE4!Tgh$1`cx~1}+%ucJR2hmUg?7#~9j@t1#ISp(fHZdzNrDHCYy(gNW!nMxcd;<;0Q}0f-SB%zjg!~!0?oZNZUl|mAaP8=7R*QX@<HdHveBe|Gf4rAQmS!l*C^7VcL2Q|?_$QeC9qYg2Ny9ra`9l$0MJqbP1W!&35v~orb94Pxh~l#J7(OpCS)xA=Jn;^N0a`6oZU%vJkmzcJN^gc+DGKb!@ew6fn?Ppt{1a7H?qys)~UH#G-Vif;mw1dLBz>K6jFtJEihqZ$)T0SyV4=@$tIvmy6w^YLc<&t<4kQzT@R6mi9#CCG|wWO<Y**SodL8g1SMOq)Eq0}(SuuLlp&YkfHp@$$X_<%B_r7R%d;vv!XcAy1u2PY3|mg|(J2sn^qyv$gB|NR{Q31%<XQtKPry$kKR?UCo{m&SUa0k_s~}2|%MpCs+Dg-&>Pf!JvQtx47}+-MhK$@@S!pd4USQb;)KoB(^}NuSyxgeIN$A58)5BBdY1RvKsc1u}lU<|hGJUR=;KNcTD9IgYX$)B+0Cz;PER=L|<OL#417nd<r-LmVPUz@dIH1LBrN<n8O??v%_U1T?s@=o;el!*BXRnfr?SySdd#k#v2sbN93{jew)<23|^C04MD)mXB!{qCo6XALnRCNbos7FgO3X4T`EHdMiO7{b6G0!xMv0{773D=L~p@L#v0$dEg=<L<X@~7+$I_n0uOHNl)YQ6e1;}->D)h6)3OL)r5t|Jc+>U^-)hz$bm0Ut>#ZvYT966<kMq2t3o0RMsF!9*8tIs@x&HP+qHtS7tFX(tTf%^MHXTiOG<%?wFlktCR_$s*ggV!J5zlQ9&h2}2JxA3JNMKJE+z<(NZ-U8RIasDamt3MLq+8oNDzs8Ho=X-<gfkABDz8esAaEKy5U6)@xqNzMkqS~$=ki*UAa@|uVrPvKEAwg^{6Nb=N~93muzM!Fq7N)V`oRHVe9JeOx2T@Vb?1ny8TOJ|&z<BPe1_*1?^=19j-OpXaqBo2F2s1J)7iTmT+KuJU)5H36eI<DR*YV4XV6TYE!<wH#alb3KymJve}x%Zl_`k1@sua~%J5?Y=bK?C5=vO}&=C&jW{!eJ$Mi5<g1qsxn(!uKXPQylQq-g3zrO-lc^zeon-3X8>*>i3wm4akj`Url0T1mN&uL=$FzbLatN3_CIhKF6f>8TwTyTA!`KPlz|yD@_j*k1|CLN_8-EMm#90zgm<s6BY2H6VqZe5=pX!?S*Q}PP<A}xeIEZ6bc*B{5=MV=p-A?z3C(er74E$xVD#Mu_T(sf^OF<8(?;U`}W7`VUUBZineT#jTzB8EA5I61><s2v^0}s&p5%u%1@_BvK+*qCZ|!`j5P{XcVK7_vW-*et_q|m0xRlfFAbdLFH887%AwkfsT(2Q9(2`>8Zt2uu6W^6)<UgVlh`EMHnDJZk!Zw<+#ydKwM&8&i~)6pa@EfQn5jjRDePV2Aozq75m(TQg1mrQrqP(1zkk)Z^$`NF`rbGL1-C3f=CkbmPEW}RD#G;8v89DuJKEI9=<ZJ;;wbwiy1IoP4KK`jF;ypyNA`QfZl&F<I0+1q^!deMeUImVW+@*-y&GLL*H+r7)dobBLaY>?+K!{<>*3VY>SHTX6529N<RZFcAymQF$`ph5?x|PV0QU*Yh;E7?jZ)KE1;l>QORsoIGT|zLkqv&JG~~eHk-YC>d29}LvP7I5T-2Nu7_MhNe&xk9X@sT};N7QUXo~-%RjW}#(F16cN=ixMif@kN-$dTbIre+Lb^iw0QQt&_5@C;G5U(Qxo<%3sq-3yvsKMs4q16zi@|*|`j*z}rIMF(&tc%PE40fIrUerjF)>7JG-Cq*NPA944iSSI998mAtxX|Q)q_&kLEvKjzwP&+_SJR_F=fSfObl=Qb6(6<gj&xSit>=q@T^O?bZ4!v3@xc@^Thl^cWr85#{%6~45bABH4v4LyxHxmjx^n7@T8M%cR!|X?rIS%lb7tL<XDLO6c9S)4<!gC$<cVHm-iLE%(NEim+)B~m=E)-^`N2Xym(E(_+1U~=xlg@w>D&*{{S64z8!xH5a6EzDtla9r34$S!XKB=O&@^2LbYLDX!W)cIpN44hqTZ~9MaM2)hK68u((vnyfZsP|YXMgg0Umz5X$z61&>4M#WOYTLPoq$Xn@IpANA^$2S{jzl-&9{hq@cb|jzPN)etRm|xQ=N3tH#$l{Ae->Wt<63%{N2eFDU&1zkt&LSTb)X%@!y$)g=PB&*(7&0kBXB-BMz{3&kO8P>t56Ev3+c1-OwsZ7qa5?`N@QT>(K2d8(a=WBZW6qKU~+QC4Uk_?;f*%+}J-(^biB`55N&DUY)Ko<g16>o|;^>Es_KpJ_C1rfPkW=36uKd-o#_^j8#q)gd)!leU&J0V#AL;loMJt}a`NMzdY{p%x{s3&LMp*8v`{JcLb)19b}|QpX1!8?;tI=;_(*VqT{_UP(@W8flIvPfN4jy*4<(`U<|R9x7Wi)09ZQNS1y~78Z!zQtcL&%|Q|kY-NL|z>^}}-@8JR6s>4_#`OU0e?`_5@>yi2kJaYosB#Gr1>`F5b7z&wUs)l96#~t17QFi!oDjwa6`W<5rG8eW^LaVy_c9{{+%D{%>?F^l(5PjuM*Ts^T7Rzls8n^}e#?HuGrG1Ek$rfgO@BiP%hM7FYOPhH{G*fW-IaIBY6@_OAahM<&B0DA0XU8AaLa;`eQ5)50j-TP3BZdK3bCbBgnEU#wPRc|?nqK`Akk{}2~SrJsiR%yVR^iZd9NgAyR2E2m|KgBl2#jI+^F(~sy=_Hdn8t()RgT@#geI?5ZmEnLiQ{Lfo>_`@;_XYne<zs_Mt)?A^9a-dW@aQjJv}sB-ZL8*6Kx2%!lGY3SnVLVaky#uM{13aSo6ROVAieVK1XyiDWAo^f(hCg9KoeN+j`m>{LR8<+B<azglZWCG7dRN`t$o@+D}<mcVNSaN8Yak6eT^S3s2sUzp{ITWN!U8V`0W4+dzOPiFDck!yK1tjeguC7FO^FkAXn^^u`xNs=h79CT#J%tLLeQgc8PqLG3uay=NTRuoqtS`>A(O<_OOilUB@h?G3|BV}Symrsu|ODFnRPf(-P7#fMbx+6=B9^)1&Nm%Wk3!<ae8rL$2>phb)vUHtWt<oT1ry0|wC#ST*n@s)Inwv1%B2JB{Ipn2l0n;`Tg^_{S(lpJD)RkbQl6?k_O4WX)>5-f;fjbvxq%N$KmUew&PhMp0jGeEgFkgfo0B$sx%0u(&1%C;w=sqVyxbuK7CD;98E|gpwal}<{)y()=xf%fB@k)g#+4T>}=(_{$_;VdgfHakYwHzO|ln;+;_RV+i|J-foT-7s9#FFLgxZ<Z;+jk=)>XJ>c2qwfwn=++}PRzcu+sR~Q-+ICTR~=+2jpilPAubn^c^sWmzt6;!v38U5l-~^xb13<Nxp;1u6}c39&a2W;FmTM%5za3q@7_%jrVCW~sJMK45zZ-^`4s;+fHRk2knOCo4ouakE(`fPLqBcq;Ovv*^98M(G*Pt3Q#~uP1u>~5v%N)*;dttxhi`)8_Zwsg77Ov-Z(K*Iq8CbYfnxF&3&rEG^)QL)DhH5tVk4R8v2hJtMpMj~U>@>9QtN{DPs+kz2cjg&jj)W2oIwr3*Fm-P)u`U{0l9a@a@(|eHnb~Eu(ja1XX%?PM>-WO(Q49Y>OYapio$C$0tixozM#8gi21N;UZ?>{*f`_R34`t|D0VWVA;pV`0b5-DMit%5aa$NiCLkUKbi~``ShGTpHx@Jyojd7CC~sbz6$(V)4Imie^{H9*;pFLEM{=#z!>TUJqwF1~37yrRODaVL8A4OlW;fK`R=SxNu<c}fo7QFw50Z$1G>`pIr^N-N`_e5d6~0I@R+r_lO-IE~z85EE<4#&i#QDx-m~S-}9lHqN<3C~{s6(l&<*|^v*3AP(+Pu+c;fIAk2}RlZATt-sW0K62+LhO8Ai-82laY&hSZd{!nmGZoNEreun^dhBRV%~7&%4Z_)@9<wj&<s7aS3?Jx!L5fObGx(D#_|{p7aZIo&eLv#%7rLap|0qt;r-M&?cU+UhKL9{z?;Z^~wo)5Q4in<KRnT;xkHNBpDJ_sM=UHe8Sjv%@HHn&3wS~GF!k%bhpu35xj<yLps4&cSU(bJQ9Es3@FOCkYJcxPL5;cC62p7<z)I+sG^gI-ZRml&@wQNzBg9(Sd#hX0x?*0Pv^Rw)$+hHJc$aB6COo)(eY!gB9xXu!f2P^Z^l&pY$A;I>-o9?huaJ_<)eOS)uGlXPTH}>r%)X93ri4#SHsLS?Xndnv60ObR=R0{^pr%FRsc7mjHl1eRaj>`N2n<IS0==Z*tiNWMhTz--_FLUfDqDlmHc7r_yPE>uP?{0@AqQasH!A&Vr7(pnwP8V7kweCSyC?4UP?ACa+m7A3?^2PgUZc5FA0^)Rp;5-84xz)wiuInn$2)<S|oW|Pi?D#5qf5zm6m!h7Ej6!83$epb2EJBk<|&Trs5|zL0}bz3wTyUt+FXKT1>JyS;oT52haO)fr;Qjgsy}o0-kJ|FHYyk)O2teHVfsz)yfsNYZ>tDcxTaB9eA<D%?U+NM1+f#Jfx-E>BVD3tp;UF7WS@Dk2ENAyw!-Ei_#?eO2y(~2k~%gmoJ@NUN9RHDqSYIaNju?mwmbGGS!{IBAV=2rQYYd)H|6gOHV2>jlJ{u!i#n_LO@K)Wt0TskO@{k$k#P1E2FWBvkF&lbCO*SkO)h6>FLNSAv7Z7)o-qp3$M`=`L@g1b~1ap;=5KCs)*>dhYGT<c=6KC;*L=&!w8}xZ?DKfVz<xW6URnB4?N>VTfy-%vR<OI;5*moBX?cFex0}QJp2$YME%eC7VBs4wb^o#UNb4B78l|h&gC--8SsOQ=GPV3ds&m5tX!SSN7%<TQXgZnONeGdcOR3VON{U=cam2m^@ZM;ZEghc5l-8#W}dl5H>FEFS{U#v6N}etjvx+#>Nk1?wp|(=(~2h*f@$79Bb6j|Z;ebON3u#=f?OP*Mxjd-tMeL_$CL*U8KsH>zAA$_z4pa@27{8!kYy(^7xpo8;*uh3SEX`#h^(wtiV!O}aHHNPZM3V9vqpBSZmW6~;qp1MZ61JWs$W$zpOLS&1MsEWcel})hy-;fP>y`G^O2oSq7hm9QwjmsFHIRGml0FvO+E<Eb0t?%&`Jy>>OTaD5fiahDXsHrGBT@@D-}@iYsrRa|51hT$U!Z!T*PA<9s1i-aR~HFq+CU1JXSVB;~8qYf5G$%R8x}rxQ@<%j-^4tNc6OwP>nUq%}co8`r;zV0^sMqm>9;_z9PRcNaA*>?th!6`8<5*4Mw?UM4#`3z4>&f9+keCh000LrI76sOCA>3ovP}eRo;Ns>|L8Vn1a8ogL4)Qv{<0TL0y&AHMW#U?$GoCgRMI~eqH8ZR%d+`1*&qCsI}@;a2b+^0^c#0SUES2z^v&$?935^TG<;VHcb|#O4Co)Y+iloIsDstgrZk**~flt&8Ze7r4>-Kxjl?;uY;u$S9%wR;YuM2u8mb51n4h>G(Sq*b>YcKFNvVd7sfgd1xw_@2m+Eh1pOx{7wZIu^YM)EGGl!qfU~~-#5!CnCz;5cqnJS+vq+4flv<vX5y<mdl{89Eg9PTD;67hYqN=;@Vb60o+ckLUu%Qz^r34rnhG!RuW~@1$cEXTwohChuOQ(SYjN2@71!Fp-$o2Q@3oI?eWnB&nXgqK2(#vv@_<T#zfOTKk4&W2GsYLy;ss1u#zSqSFa!~~qgt^eR-S)Mk>WP<`a8a)j0)(&%*yjZP#km}D?5|FRUXPD)b#XVtNjoM0x<as~`Qfqnn3hi17)~jpW?uN3(`|&Mo!5djmlVxN@*EjFVChY9h3tazVg^h&$%`d~=CYJLpyvL|3?iayP&`F?v>?P7L06`Kr`VR1Fl~uUw8Makb?P-?%W>&dX-THod0W3YkasF%Yz~~l(;h`k8W&>%IDxtntJs_x*UPIu1&YIVD5PxUCC(~hj;zT8wpY4Pnr#=4JkFkAuJOjc&?F6aR@<7()S1O;6%EUBZBioEN%Mon9KMktE0tLT{)b%)sofMtSoo}>szWMMQCvGOk1#}+lDWJf52EGkYn3#(Vw9Y7u&nQb!kr$~URA5Ej}fShFMOeBY7p)s)T_3KD(R)|%&PBF=g^r}ewLuaQV-42s=Q+GMwe2wD6XzN*jarWQC(B7P)J<mxsHbQW*x}>k6qhFF&7FeP+_Q2@g!j8GWz~@(h@b{-V=Q}NGNg%?!b%x2Z_%|6ohoWWYmQ*nvm79wy{LQu!6jKfZu6+2DX%(IFDso$9wQ{iuD#WZB$i;wr8d3G^*a3zPSwKsz&_-o{}gRkE#}=Dvm3I`Xy2Pcxg^>Wbt^p8ZEpW#NLD1<sMBm!SY*$hEhcng@*@H@ieqet9U}9Dyu%6o0Phy@;B+6=_h+u*0(Gp$Cz|{o>B>Y$7jFlqnG6;sg@@ru$9UiqFC3RWcG0`P)wa6J7pO%$@)r^WbB>L+i{dvY_6HASVzQts9Xsx>L@XiiaGyhW$G1!tf@AiR5+lPZ&!<JStU_L`4a|J7?rawZGO$dsXl6hT$HL>%#nQ4kt9o8r3UmN>7ZhggS6+;^?2HW`WqE&_0>zskx$bjGG6=0kp4o}5KFfjiLE<$K7tsxl9(_pD#l<nO1KV#)nk|ueYCYVl!L#e&@H>kZTe>X`c;Ley!V#@=)^o<w9JHLg!P!xEl&1s4%9`)HOkw%tmY>y`uJZKSXPmiQ>9*(%R#)bq5Y8AMQ*b1g%M>`mT-R|4MV7IqcTgeK~@6=b>T_P?dAhgWW2080YjZ)s+4Z7a^0>5A&DthL5mJE>V_136hJC3r<WD@NaB@R4cqj%)CGb<%ZJ7s6l7A>h5@RWG3#MaZR-|D$lwrKJe^ki?5t|Zn4U@UCMyoqz{NA8*eWHev^v?ZXnciA?mTt4k~n)W5I8||fXUR_g;q37Nu|i@dud>&lSbi`4cR@2KuDzg$lZmy9Ihk#m{;M}Fc7Ab+eLz`CUeruhld`y(`IV@9;r}nlny{t7jg;1A=<vwuqBRemWiqA7(W6$BW8q<&(dNu>`0aM0wWOJ>D)e7vTwc7Vt&*-A3#1oT8ZLBRu_~*Ily0It+hXr99|QA{AP<8b2kV!$tqltu%3$bMP)3=NvusGE|!v=7QR-%bw*38jG2-`YgVF=giMO1cWF$)s5VumWdW_N){UnQ7q+|y!GBk^LdarqjQk`CKN>}pQ_30x?~jyjiI)GdW<;sBP0MaGci}z{4R<adozW)b|49OVeQJuUvVTLUN#ws2ij!K2Q}>*cZAn>XQk6n!@l**&#LL$}tVG7Fcf8wOz}bW}J^(oo0jGgz8j8Bh!>d@pDrjJGq>{9Snibu8TyDu(tqQH8iZW76-q1ZED=!yhtZf)Eb(vFkC_gPToIEXx?(Lyf7dNU#*@e5!+)S1l-x7EjZAamBmEl)sT`^IFHqo+8evb>-4b6%bYeX>VYD5hiR7`nL&14%VdXDQylx-k*x;%^Oy8@K^vI}(qeBAFV%h6ELmLldg!7O_ERf#B><pjNkm2+H<$Og=AF(~}u!7>tKgqhn$2GIPMt-`BR>pZdVX|TOggf{_GGptps_$cMvf_sCBm(eRFtE2}_jRKC5+p87@+FC~H%!$s3e1n$iz)?_M3jTnGTFqlbev!nUMCv>TZ(-JwOaUcZ|GGHO5>J-qin3{L&cuOhFbmX+vQj46;!bpS@N?l9?l)WQ{BZs8W9W-K{ohQ`eET6Y<ab}1xWA7db+!#Qq<!ErTt{jfY0rMl_Q+Dy3h=ELXc(U1wzac|7id}G;k8vqYSkPvqS1^sW<bD*wpvoKk3*4Ed@a<I5MNAbXEf%Xad2~QfQMO9bCHTkm0V_q1ZmlJd8d)gnU)RCm~n#Z#c%zLTb&Gi?p`G;{GJ@~l37k(-JV!T{0-eVaIw$K4C}>cHi2!>S~BEp%zF#{2J(*ZcC}s$H!$lgF`=v%@BX%ZYi>6Tf2C*`_Pi7rYPD<Du)y1nwhz&Fim#Nq8Ll+zrTbXT-R+0$1<@x&hi8q7+*SM^RGNxxv3x8DMAx1eOrDPnM*2$HQC>{U=<4&_PE+ff+1`CZ9%f%w<1gEJf0Ng4Jd(|QI5rRe3!QmfDg')).decode('utf-8'))
_LEGACY_ACTIONS_10C4S_3Q = json.loads(zlib.decompress(base64.b85decode('c-rk<O>ZMva{Mnk>mX93K77-3bKQ;Aj2cq+66=957{F^7FxH2$Z-)Q7dnK}1RWC9!GV@W=jCEsE?5g+uG9x1+fBv77fBW^fzyIyGlYjd8<cDvcZ$JL><>uk*xBJb><LSx2|N5W*`d{Dw^8MrAfBo%0{`TMBKmUC4>GRWHwGTgh`|B?^KYjl3=Jw?D<ip+e<aF75{qSkC`7-*$!)EjG``6n~o13pEr<b#@f85;O{d97=7=Hfc{_f+q4_^=e<Kpr0e^199`||n2pTB)MylFA&+s`N4&BNEHw*GW?|Mk<;r{SyFhv`5(Y;JE4Z#|#Cb^o~0t3X4>uRVO4PX%hg>~-es!5$7Rd76{Oq_4YQk#~K$z4@@Q#uN2t{~y5HX6+_#-TjyGcsA{L`tGO0Vwm)GH&e#X+!5Z~%-?@l9yeb%_wz+G|8BZ^;L=^r7tzDrxA`J!7w4b;u`?#$%zDSBvK^f10MAD0(7z8iyQR7R(eut6bv-nfhv8~px*vt{uiWVZ`wvYH*a^)FCU4n|Js7jma5OX4{zjj%-MG`Cn>=^E^A1DUPLr`N7sBBNHiLPz^0Q^q1#M)}q2o{9zNPwD%HR0&2!?QX!hkvQ=1m{O;T^+=?`Q7=`Vbqq!?;%-y!$1c^uEuh6W*l*`~N$7Q`hIZA70_Hvs>lduqK_uG;o3RdFuRZjcnf+Z^7IiAwO-*h(0a&aCdvV`SA6ZKW*;6e!l(uFVi!j)8M6F5?CVXcN}RB_P6$^J?0)79Ff_NjjMe97_b1}^!g9X@4Szzym#x`e?^-Fn0Jl&I5NV)!p-;@z!-sh0{3dSv_obx@58vaULV~71de^cAZ4x!{Nz25jRpGTK9G3?qWxI#N9`sj9VmNHCEHioK-4$)=bv~wHP=@Gp4`VlZ#m#R0OS7f$krJ2H-8J95ZlskU+8hJsY-CO7dEWlpVt3r^1TmisFezG=M4kfR%j3BDU80D!Q$Ufz56?a)JVr6yK1FFGGjjsZyg<2@w-!Od!=)sAw<Y}=}w^cYsuK47j0%(xE*6ckrAiKYk$COqLv3S84~swUGxXk&&5UwdgTm;4;e-d-Z_-@#{sV1AN&5;-{E6*0IP?oV@KX$2;YUA)?om}2+6naZai4#&T05c(rXOhDYXEkXAxyUkQgczX+MdoR~=b)!5d@q>E`|~R>%6?_yM#)jAEmq8v2qPqVZT%C<g7|v@ytmOi%(+_@ECRdwy$d(2-F!49a+<d^iN)E0bk+Sfl$vIYd0=K|eeZT{YwPjSO@hgPC(Q=zRv>kclw24{m8v?`FgMW2+$1+OwQ?zq@+c?yWJk#>B@(#I)KOF%MtwZ?->d?(hE!ESVI-<aWq>JEUPQyTeVafkq=1k4HeGUJ!%|-JJn5at=jh?@}9C$O2EtGO@<i$rx*jAxs=pDXov;!*HehkJE6r{f#`@w3z(bG1OV|PUa(!Tm@FYLtj5PGiwo|PtOg_gjm~Kju74iu6E>c8JO^BbiDV#)@y~yE+W{`x@gy&KPHY(AztR#2*gl9$E)H@Q(xT!V=~tYLn}s?;P&qB_9-m}npV3%?k4E_`S|f9ZR;KUdEDE;*V56cnS+cX(TTG%A8PAXkPY7Htc2I{FcBgs2V=>X0{aJKD)lxLQd7kI5G_8Y-q#WgRicOKzDtcA)kZ&Eq)b9@n@?%Hb7SR4#F_|T(|I^H)<lFdjyO}(#DelJ2z0)kZglj`Pm9bpV8e_)>4O4Jofbfzr_SUUU%)4snYYKXqAr5jvWry2ZW<d~`IZ;VSPh$FB4tW*bmj2Yj8_o@L{RoF$Tl_F0Ipwlo#|+W;!k@6WPIJ;<i-y<q+pyE=aI>V8M}UZcFaPM?roZpXH&ftNPz)lLu4s{_JR4oaWbkIaeN3qk&I?1zqDdKupLAbO!a(nmctref$%>R4PcX9e(J;{TU5}$Wr+{L_FKDNdf8@e$a%i9;0J%CnLRP|0vG_Mtq~gMw#)3fX8&R)4+XhN2b;h%$imGmZRM40G!TrNJm-`afH6;;ZIN*{U~4e^nL`!~sLu6_4?j!k-#=1mNd~W`JOVl4`xSD*`yjcCun$sa0bXNoBP5VBCKR?d0sL9W^GKv`fZMfZuyxAy8?LM|8f=;Vz^Eh69F;%PghDtmmD{hFrIgL{2)G8|x*R5WZ}9&3`S#Ddm1p@$?JxaIzY0jc+tSbPTZqtsgshL_HLK_pgGCpH6*zcXar(f^hLs4mydZ89%FhXgp@h`aW6Dv1Bl}>ckB#95C~2*E4vl0IdWZ`%K1zJjshArT8q0GZFm2s7B!MNaMSOVHoHH=jEo#)B<478)m0xR((qzoFrOmQShVK>ILiZ`SF`OB^Q~S#NTh2L=9N)x;Y{I$c8kxRXuhx-mKxZ%J(dzACD;``}tz*@9c;K{9XvLjK;uLmXX6*}FNN4Vf$2{eNv#0@2ard#N!*)2a$zfh;Rh2gek^0{4ky6d-0YQVG4Pu08M_2H|SJsI_ey1r^9m72bW{2mU6MAkevZ*`kuDi;t7%@ZA&z1BoewasutxVaI4W$lNk}#Pc->%|A2&?^(o=T~cb#$HOXExXhuOyIGD8W3`d#GX%LyPk)s`2$Q9R&sKp}uf9s!P<rca4fT%5t&@+WML<NFX)@_-2h?3>tzQQb$^(wR$r=KIg?!<er2!i5f?q3I$G`CT@;~7VJ!03!6Q)>OpJ63iY*H>o`F-6G-AL?-VA?gkmufDHV{ULVN|tsz?OE6uUVgo%RymhBBE)`cRz47id-f)Oibo2u!fI(5&~tdf%SMK_+CMKkfU1Iw)@@Y~HqwhRubLb<EdA-O%k_C4w%ulEpzX0i?^vn$U4FHw&2lD#sxQs;8XL7(<-;P#HcOAV1QW6A#y+hm>^#AgjW*$xoHUpmIT;)+c6IH*GD$q;hk~{o3(c#8_)JQ;R>Eoj17I*1~kL+v|YG>*T01axap|B+Jc!(R~|>xVva8IvWU&HB2xB?OmKbCfKtK-D`gY@P)%yFzAk(dew#~R{kYLqo?qcVkpE$5s{FazcvBO#lm0nY#q6>uXtV<jwko(;xQBcC(szktlD$k#?=}7t**&qozAuUWs-?Wnn_bv31N?fplJ`hd0W`4aUIxa(ml>chD#21?)G_Lr;%9!Q`ZKnJ#!uo3&>)YJG_4EC?-X-r64+NR&zvi*oL@s>hiXT(ka(6LhLYFbw+A4AP;QZ>%>|MO)>#$vz>G8uc1_ja^Qx9^QK2VN`^o?X+)(axzdA%Nj5)vB7B$5Ti~e7>*g@Dv`yW8b{%xnm+Zuy01;ZDo^c4vhz9-z{N^em9MT5zAeFiM3{`b-4rvmw=rZ~;Q~qmmrs~(Ex$1KXH;i)R4LnOSOjYBMvRd0XbewO)ZHpOOGDANA4&u-OZ&)?TG`p)S8CcdJYy?JpeL<M#j%Gz@g~WIs0_Ycdh^tyzS~i3YL9=Gw;)-E5G2O!^VkTEkSOic5rh3E;;=rQtwNCBS45@>RF4*=9FIf-Qf)p9HxGZ5ZhkHzLZMOr8zKZb>eq>~QhnQ%YqCn=OA*6R@`7&+&T8oNx&(EvM3Zl4+Mr$f9%n)8F%ScikmwnSPKQ`v!JB$Uy+{e@Q${j4JTI(l@ILxu0Bw3?{R?R_>z&stPtichvf^op+0M69KC3<2m!<XasTsaWU@i7>{cu<3-CIp<bLiMbaJVQprUKEXqvtW`6wU6VDI*9g%6bIS`kmMTGMV6~NPHY9~SO|!Ta^{X7JWMGhQ*8;g(xx1Kj(8uT(!qw8p3b6T)9z;9Fd~KP=haB}ipX~$X(eDtlRYmE4ci+nHKLf^!&qGvzgNfs4TumD4dIcuL%IU%*;Ug2F-OIYRN@@TSstG|XlhiD5uLOED;x6JDv)%TwKc%XKQL+)=9wsspB=M3;R)uM$DIg6a5Kq-W0g?3Td%JR+mF)~u?x6$+|d_MjKoMYfs;HKp~y{@mx4H4R2vez&M4I>$>5B8_l)!~?`W3^M=>*tTp<<DER0}h*Z#XhhP5hw7w-OxaV}bG9$FGo;((1&B)tCC7RmAoqJj@qd3H6U0-0hmp&}QB1QvhN-*=s?%w!aSA|^FK{jPWsP(2E%sa@zkt1SqZ1oholxBQH)<X|vbq3#eOh^|d+ay!Afofgp)$OspuASmxksx2Q;JegiL2}KA;P(hMyk?5sCIx<@xJdPjJig4SydQ23SrzgJ{y=1#z^J&?kKtUJE@t`SFY5!Z&191ZCY%!RrCIa#ynAut$9$gH4p=a9~q*t|Gg6fLLUG1gdVzup|ZdLoC)mrS(Q0iF|QAbItu!n273}hc98To8rMGrBbX~3plsp;S^oi0O6#!*_l`nu`S+-i<SOo+6$B63i|_1j|!@XQQ!^LK;<Fmeq9^BYW~R1+0AcAlLNgB}GJdK(Pf*9$?E@*0vpyk4E6Ec8H}b;>B$N6Uz3pUQj<ma7o1<DtM#Ds5nqDg(pNhCK(p#Y=~8Lc@IVSjkU`d1;|t3z9W7wtb+$1DOq}>VUCMrZbVD8ni=x7>?=Fh3t+T_+{7M%Q)Z}Zi=LHQ!9k6rU8j^nVG3(8X<5YIb?)P<{&NS=ZVo!AW;N3mbnvMN_z@Ryqt%kZ@N~GNIllrT}r7_N1LZWv{^+QeCjRRjGGu-_N<${SQ&o6Wd!Qs9KArgL^$0*qEG<9i@9tsmtfxnb>C?0_Ogpt_M)NNSWU1(F8mRKxXFruF;2F^X7Pp=JI&NKC=#_nq%O~hcXY&ff_sss-L95twZdoBjeys!r|UD}lhM120U&v82MP#Cb+9XhQe=-%{Zvns&KreNu_<4eq18j9eh8(>R`&pi_~>~DF6b4mxTwo_Q5#;<f^s5FStC+Wz!Rgc()e`=#Y%FDv4M7Y;B`tgSsf9vOeYFUIn`ZPhoou{@__iG)PcV&QlxpV7?XaqBfvtVPs_fF<`C29Kl2Qu9{qK(?bMQ_NJyGeeVt~j$9h5nk%$d7?)b;g{~YPjV%fXxm@WQXDtM&0c41v!))Mo(vL0EK&8vi2gU*bkt9Wqm+o^2+G<<{CYv-jnZ?;oSaoi@rPDiwhjYmv~#Fy6U4r`I3?rHRJEMgM9$3S!@5bXl%Ce9>g7sk$5zSTf{j;>z{$#X4p)9~&WiTBcNH(_bOD)HW}Gal5iJx(w_UWhVT_pl_jKs~*q)Y@q6Pp=4Rh?1OS$Ft~==WBN=-{CDNU;#_%<H@{rje5=NMQ(bOH)n1m2CW=!gFYp+pEF733*Wvccl7X(T2ea8-_pcJlF9;7&=)PBF#iGdf+`HKt<AmqQk)#2mjSq!#)Q48B%#Gvq=fB4qEAUU;yRZr??JFk3^nXdU}^(`N_z<4T(ig=ehsAlEk??GQh<Ujo-jS`JmNu0fB=il1L&*|6(lAd$(1BbW<)nHoz;Y7$O`8@gnibfh%#nn{x7Jv+BW6QFcttu`&AhVz^{-g3bgihouUBJN;BI0f*xMgy@igx9|YX!QN*H%lk5U}e^)ASiscCN?sHbdtzBX*9X8Gg@Oi8~3gl6N^ldCy5nqi;9lFywS?@BE*6QePc>+$3FInLED2ix^%pL&Qs3YU8NnIGpcYvr4ZYXxcfd}%}WF0;orQoZF>L-OW|8>iLah1ueM|g%=fa>Ci6Xe!yizsXYCxis+Ii}-9rZig1?%8X7CrJx1NFWjFJ%&mqUW1B>qL)F)8lo5hPi$fPQ9dYEf_Eo^Uo`y*{GQQe5)FrtkH(<T3QTKzBa+Y{6cP1{BM&XLE_L{0hCa*fhjmf2n}e1rB-qQe+n$+FnlI4b%hse*g6D4O8;?Q5maIttp}wfs;pS;-nco5`LqlZDv{FkdWdJdA8RYuN6C6T$LyR;dl|O3i1QlMIu}WMu2BSi0W2A@#*OpBZq_;X<GM4bFQ@+r0W$ZIhuX*{#5{ceW%?A!OL0T2+(YYIS=$Tp66+#N&e1#&La%qT;WhRAj$}tzD6eX4OqUDAN2-;B!3ouAVFHb2~`OwtphGVUiB`yIk^6W_T(7_wjy;=}Z$izlNdm+&SnxMoa5vT{KR#Z%ssk9y7X?iIhV3~K+TraugBUZOU^N!<!s+!^eK`cOYiRTijI-+4E_+=pbh&N0VlTi69@rIK%qNehJ9%#A0=|+<^G7U=L%_gfxcTHAUN9ZcC3MlR0R2f{^-#klWDq2kvwdIWcFoI+Lv4s0|N<;+=MIeoj!shUKb@$6vA`6cnp0thuE3BB_HCKYFqLMmAJ$9}QqiT7;)@SuMsYWT2h};kWZrxe((t_sJ>27C*F_ny@Xz$J&SasF0Uzw_&%7^1t{%lf-W@$&S2Misq&bLtv#Nlgk`fQUaH9HbJBS90Sr$XUU5~-3LF8*k2wM9r=qOYD~bHR3Ypl~IzAg9u)80=K!621H*4^N=s2p2laF{Boc8FfbD6bJIZf;f>fYo~>*3Xn3?DrHG|TgWtNZE?<?#67%3k+1+a%q_Pgsoii+ga$*-<9M~wDZJ7faoTaHnkrd`EJbUQ7!^ly`$UJappV&Yt)YHc+zGLIW2W%RslN8EQp7N|iU^I6)s`NrVt1K5`xxB`8z>s&r}kxCnla$_i%Zj?(lc3$lqA*SJU3^ahM)nh+6-~2fS;IU<8((>x1|gVm#W2(%tBq9DIw*_Tb4Q<QD5Q7>h}NhlFaAlRwokHMOwg#$2T-79u|e_L$fOgRH35r5}>%%!~z`lSy$58WhAqYk%U-^pHvG`o=nAzAvvJ2-e{U`LjGExe#K;jv;spEQl9@l*^OhRX+A-G=Vt=>t0^Ni!~QP+2sM7*Sk)ibNbEoZi&O?mND@&yU!Z=qWoER}LV7)!&Lw|5=mwkvrSijWJCQ~25vN{3If+>_Q{T8)A<B`h4wS%v8+j3&9$$5u3`Hf!SSq(|ktiIWPY3<U(qLxotX+yiMoz++7WkBM52sC?|Encp<=9?qqS>?CEf2C}DH>JM#iIb7S%OD#e4Lc7S|&GTt{77+QPh&vn*0nTeu6lWzm^`A<NtEcR?FG5x?0j{l%%VjGtc`SpNg2e9@TLs>@jQbv|>(ML$WI@jw}`vr-y@0^)nL-!!4&H?<=jajcKEgv4e{y%Tn2TJ0m~^3~LIJRDqnU@UuomOT(4bhiQgF+dh+IA1a~|uOk&9CGjgyDOx=TKwhKnm4#%yzPi*K11xy?E6P=}D|;PBYzlIfXbZKpUEgSMQqGigIa!AaAHXDNRn!*U9-oxdyTG&}_X`(Gw4_y*X%tC<@o{Tb6U#<W0zioFYl*$81*;|fVo0Y;<~4>YZ5g)b(E{7>_}dcBCNng7s+t<2U!}M(4sjw=$=vXPX=&J4X%T%-s`sbIv7!R)iOIUG5l1#=zLi#<Oh0KAX`Boh;VQT9)LLMSsb;9e&}Faa_M~E+Wa1=8)Yzk(bWxFv6DT8al7tFG47RrJz}CSF=tYHC1!)CoCaXljivUxq$Pd!jE6)&Qtif2aU2+z6E-M8)Y$(fj1eD*u<|OUQlW}q#O+b9PS@KG-?$5dJ$LRf?H~t!bNvfb5?i!)*c&Fr*dk@0{*HH$@P_o9qHA~7X)U@4nNwA)na>5k&8Bpr$V;xoet6lQ4q64itAZcaBDyh9F1Im=Yn$%nKZ0hCNAFI4te-e~fYK`-RUV!w}p(L9x68N(0rz~+vS23akacsVm?qdeT%(x5No$#bp2MSgf4)B@4;q5L8iJHa4Fxn#_w^W$?jB3Ovi95Mjr*UrLgC}x3l4Pb(&_W$h2Jw`Lkeb7w!Bg?#M7z`ZJaG5WcKR`4Jo+Z?lw&%1uq5h(lOkn{wI$0*vVcD@KDe?PWeF0`Bm>>UVgwM^=fx#vGDpqop%trY)!%$>&?zqwha?=W6S50OL(nHCJ#Z~orCMiS(h)B~AZ5rgXH+XAF&uyC3@|sTQuju=Vq(4<i%`O8#KRW}wUpQpx|Z^F2|RRwBR=LhJb{8e=7c2|N^R1%jFND=#{_fXesL$Ux`$E_RjS)qhzl<xXx!;>m|zp!@S0X295?Zu3YIDbkBmN7c&9HBjO+SJH4<Z*TOF_(0YP%j54RqR3yNgayir0!ofDS-pV$|X#h*X|e^4@)6VC#K2yrVJ4vv~&9MXNcehlD8tU`DoOOzAR6Y90g6j#Y9>L9Dpi_$kb)~HJ?S+lx`RN-W&Wwai+S#}vIf@O$4ixh>#-I<<-2zNq=ZzG^g*2#1pm|kUdYd5A)_lxC8^`S^A355(V2WO32`yd>r)k<|O>{&buC=oKlM`f|$NzO^Or*u-Z+KZG*qnU!7NaF0fMQPtb1J-0mppLnPckwhUb7-;B!h&k;2Bmx~umtzKp<%rCUx+YWUo)uPi?ocInjOmPrTQfR!qPvh+{P5=Y|A1P>_i1OMyM>RV(@4Pi`aRdsI#>*l(sJQsz%#$a^6Cd4RikpYN(W`aPAnbN_Lvb4uMb(oKXeGD2%(^1Y~=23@;|!rIh?6=7Uy#JHysI%Mnkb^_ZJ01O_=bA<W%(G+&9hwlWH&{y!+(D1#ZTc?L3DhZ$pNt?*Z;=HpVV`IxElMa@O+iq^{kHdQl}L%0QaJw0yk@7s--#*)zhA|AaXEo-D%syw|Sjn2qx@lxv|l&UH#j*wg6k9e}GkELlR1<2^IZ%mrLC)(0Jk%z{k%^7t>#se7UR!f&H#}+&;B26Q7L%GT2DmNt}_W>zFu^}Z#mC+>1&Y@<JljG-m8anyVLn58g)LR)JKkh4dEW?0=4jn)zt4tp&T(HAplw~$FrK>rCAfKA$7uZnClSv^Fu__4C#c6ZNLS^#CHMdesnIN^Bf~{ixMwOcwrHIh0kh3Ux>as7&QkIr_c1k(RQsTIY-ODU87YY|aiD9j~RSw*>=L6ngq+H(FGEc-}dDna3fyXJpanZ~pWtm0OA9*&QE41hZCxo7L=|+hlwmhQ*vHeSzZ;CYbcB>sa$1dX`2C5yt2wyCshn{pZ^OP}{1C}d;7)T_OY`s1VuOlUChEh8U<qk7!D6Js*Ti|-p3s+NHL9wu9q?^}zjoJwDE2UsW*+CK3CtIEueW+dN26qu4+SP(Y%(6(`($NI&PI@w3M+#PkNL$>Hq=pyx3W2B@^4+RP-SRy~&+l<O0e$MON(Uz)YO#=jF(F{qO@OXb@iNu1(@0#-ebA;udYGF&kJlx>0HJS5vaDW;smc3N#5N_eR!+x|z*W5LAjrk#3A?<QiHbw4Q+Z)}o;5n6HmQVRQcue`T8>jiP%QJw!<(vNm2$HKe|V3YNbIAA+rj>q9oFbl16a(rq*)E}mUEIdG0)6BP_E3K7P=Qj?u!Hus)aV1mEbS|Wb)+V$cdXRO313AZWIBGi~|*g-x!6^5m%u{@m9lvb369*m_-OoQr5MGEUu1?EUpoDxTX}Qbv+$bCs#*!S?30~YW(L=?M3(mr3_<+nbopgT$GQZt}bl16{uG&k;U_<vt+GqieW6<M--hSE?Qari*WLE>W#<Jst)qZ2Nd*DWArUnVNO-L(EM*Cp<=H9ca?iHpu}lItC>SNS|YoH#%HRekyY#+5Waq&?_yO?HRihova~6eCR?eC%WZLvk+2>(A13F%hzSvXEl~mwk>MnX4!Ta6?oiPT;4V@&gi;{pLq?e{+i1?!dES&}hzu{;wvm`ROfj>NWWCJK9>>S{HkbAFN*TFX&1kVkCixleA6?v(JEpGHfKI=x3+mBeOnKE(38m?zod+TiG@pyY)r@E+qbO&od0}{Zf|^u>6r@f_7nD=p5}x8uc-}M8aP9-~DR^u2^mtkxz^~{66O~wV+An7>i2h83$H0lZw8Au(c<6sGiey&k`!$uy809Qhw37EO;fT1@RZTE5l9AT|cN3^9NvfK8+(oN#GG(%$-dYbiGPe@|Id_U!=3uyOLW@i%%sqicIi^(~<uH+M=7(q30f?HQBO%<tQx8+U2*DF=X&%c=sXwOt$fO@=kUdFuTb4^s%5<Ww2O$^fq~P#kpte+PC_Us6>^EJ<fT}OV^8*4LCB3dAbQ1z$_{2~vRSz_AGq1>>!hJ~a_!?iOr8jZthXsSx!jf)*#XuArU7H-}@C1UQbtD6=9^<XMWz#Yus7MWWPQ`37VPkSNo?$sT#2GO?FQ&$$^N*<nP$&&g16*|Jy6g-vJrEVOlzOMavMKsoYq!{yD7TU*qFpSJ9P+LnS7D+6fFe|MPziC?W&R@jWowJXj_?s0cOtDN=+<$?nid8xXt7YLTkaV{QhDq14DQlSmkt}*44sSDNkvql6;7dt4G|@byqKe(bfcMS&^KB=CRB`4tz>E;N6;|xT9N!aq45_bjg)#WZfp=8XK>!zqGYJQ4hcb&8Sq2p!TE39%t%2|%>@dO<%riyoma0_MwUTrtXS>$mIcJw(_XHk&%gn#fvI^7zR5!Vwk4e7Eji_kylLA?1Q6Xc6*Kk{t0{_{gM-N-9<ay+cUEczZAB$AdqBCEPNWu_ua};#;3D8gnvcLkV(jK-p0nhHE*=xwT3>D+9;P4s^3l1nZwiHH0`fF18$OKuFVt%IWu}F@2O+eCv@Y}^v+&wxfdT3Ap*mUzXYwsIt8Zg+t(sdpDSpGnCDqa~6ONkX@kwq>)JuftphOY~x~Z<QsKxP3Ie*E#a8G?-+Mw0|iL1qPq-mv~v?)W?!5xFgH8R;`%NkhL=^Z1Hw|A#*(dsXDcGd78XJD1XD!?2DT{b&cd@1Y&B&v3Ot!@xi-uv<X?#mG_C4z;5jML={U6m}UKzL)uxCy-yyr4ZZkN!&Jz4v`-d;F0*@g=)SUh2zolYTN082gnjO2zu=z>6FE>d5qTcZ_HshIh{XL=FukXEH68f=dp($U*Y>f8tM+hX')).decode('utf-8'))
_LEGACY_ACTIONS_8C6S_3Q = json.loads(zlib.decompress(base64.b85decode('c-rk<%WfRW5&RdPc~H-TL;A**dM(0SQJ^Rf*1}-1fY&f!tPgA74F9`javt4Xk&%&EHRN!ilLn(<cfBh!GBWbZf6o5)^KZZY^4r-TzMOsc`R4ZBPam&8J$%0J&o<|0fB*TPfBo0j|M~jy*Pnm;$1nf>`uWS*`<uuA)joXp`NyBGKivFqeS3C(_V#Xlc0Mb<{`9`@KMwxmQ{TV)`t|yKfBkTFzM6dfLw|es;q1KI|NP_q-Mi0k9}a(UvDy6heAuy%H*f#+`QzbD-Jow@&er{>hsU;lxVwLN|M+SD)#Sr?AU^fCw}-dRr*GXoZtyD5kl||&pQclR8ZddCIeV~&`<6V;NjK{2_E+RxA8)VU_SSf!{_Oq$ylv8M^49I249Bx*$K!V&4vS&b*X>LhKXXU8zn;GTusp6G`upi3ntnH4J#gvHri<v)-RJ2dDi`PX|KAy-ZzjECQ&|qqcz`FPbm-sP>+RCq{pe|D4!Rzi%foP$FWrs8@K^41f&GUj2keAm1(UaI#~zH?U^t2yD}STU*mm6M(2bru-Fb&0ET_p>mz{99fz4nZt^8~mbwL|hbm;h#w`-|Bmhw0LJc1$Io-kmJym`|HasQ6}hp%VvC-fmUaEEcPdGPj^bkf^CpH6s_4($GR@TR8Ebw9koV<)%D+^{B{!!&S#w0Y|EY>jN+XK%sO9w9$1%!oEEczbty+rNGI>HGfv;pX<{FXJ<z(cq<D5?CVXcN}RBcDMGRJ?0+TJ0g=G8&~;q6R-eJdi@9HciP8A-n(_}ze<|~n0JNwI55J&!p-;@z!-sh0{3dWv_obx@58XSULV~71de^cAZ4x!{Nz25jRpGTK9G3?qWxImkJ?R6I#BkYO17`Efv9ip&p+{W>Rew1cyb>Hz2$)O0F3*?BTHk@-~1(TLTpRFeWAy>rYgbBp4qVe_O$j-lka_CLoHO0J8u}+wnBM0k74v~0*ik=_3rNwQX?IQ?5dRx$&CH5f9v4Dir<}L+iN-(8bXAuSKSHpeyuV#=tY|u7H-FwP-Mhu^4cFTo2cbMOooI#Mi>17^>eXNf?hd;;X{UzgLevL{eFO}Z;$=<*jM;i9l&Z~>e!KY7{YfUr)3yGF+%d?y9*DNxpNx6lJpt_cuHLW(zA#%AV>_AinO0Z)oUGDcEKBC^Zxq&&sN9U-S`2tK#XFeq1yK)IYh&;sG%5?gHy&J_hf=9AcYV5zGF{sjSV_7s`i6294Q|T0r<*j*&WvCZcq*pPkGP}PefPE@O=XVUB_Ui91VJ(fj49#%;kein$$P5;oD<tL87H+Ic<M;`MBLnV`_<sk6pyH+8QyR9`3K#-}U$RKLSf8g)q4tGT#nqn6u_^6KkN+h;DNPH0l{asL<URFe9f>RQ4{lk%cVqbSx8VXq^nPrWnG+L6xfYvH!5Y(*4J2INR<<9&B1ner*`)tnyB#BamDLR=-1EKQ%LJ5u#5|4b6mD%Ug~R-UP08<Z&68@Mv_rZ-K3s3X@$#u%l(AT{GV#j!z+8=GX|tP(jD5;!IOt-2-DX*9t=`2AAOa?(X(6Ee4ua+yC57(AV?v-C5e!JNWasw}G#vqf;{n8AYNIXJtCn)~z5LyyIC3ujTzjh@c#dC9eeb56Dz%Z78Isi1{H}d`!KsB^auT9;W*)b?m4z`spHN5_(&FO5>fql^+pnB7jY&;n-Lc5z08=OpOx@s&_%4^W}7-qi=dzWVQhtX7ouP6maU40P-|-Cdc>!KFP$qJ(d-95zLnDQW3jxY;56MUNB=dY>tVPDb3N9!&?(xMGO!@*}EXy)Mx{^e%W=VqZNuj?G2Feb$OE;Kje^tabC<LlMOR={q*dZg&^JAG$YTZdMS_s1IYG~r2yIo=KtEssAk0Reej87G&}jFHP!>$K{UZs&nIU&tl<?1|De(UHreIpPCT+i1^ru=_z-NrmFuOIZB~Yyrz;D7@JE{26Z>8O1HiO3LgQR^nLXF+U(Dp8AUEk?5qJh!xS6G`ypoLuf^n1QoYDd?=83Z{GR_8U4Te8+$btdYxt{UiXDR)=M=CAJ;MJ5zASZmkKrVP2B)1FuAoVQ3E9`BA1agLi!qO&yKMQ#tiS!L{yVe<OjdK0=SJn^>woHFu)DdTn${%S$A)J`X?N`iF%I0|lTmx`j?k9L>@cwXf`={;7v;3s;mwu*S4M@J-(ogT}MCd?5)`#($MRbb6q6@<c9K5Y@`oPQfD-mpYLEI*kpA!s238|;Yl%oPi_Q6aa8p8!pQd;vI8p$N|5Eo{6l=!4mF*hnSmUAC4E!{RGfhDd*e0bJ5XJD>N)Tlhiku*>#zg8Nh$(U<Nn`M^_-z&C-?o)7MI5T*s_Lcd6Ip;)jd=nqC3Fn?`Wcp^UT1U14oxPYxtG9=(cyM8rj#b;?fzv{v6?Y<uQ`mT!l`kkEovAAx^OP%`MGbh0yN`7`Y=;w@9OgBxs^yJAq`q}~q*Sx|fS|$81~J04qbqpf3+qH7zta?|j^UmIv%~Yw2|YI!+0>nN*Ii{+jF=(m=Sun(Kg=V-R;KL9hEfM>k}#PcU#{Xq2&?>&o=T~cb#$HOCpOpuuOyIGD8W3`d#J@A_ASn{sD{_ecoY<{hx)?ds4h|e-Zm=YD9gzrXzOdbAc5Er;F}eGF=z;KNF8aB((29d_?#C@k$V!_Bx)RaDik<%inuu%TCg*1Eo}DGS`S(qR;aICTE_{xnLrX}d8aU8CKQW_NU4A%72+#ERz)HRrr6C1>9ptYHk8Rc(uc}ve1=x#Pn}a3L|}rQLbJXN*0=3>9ArZF`P05HsDpAcVe_(WG;A(}tYf||>V|IbDiL(Kl`IaD2_UUDbwbC<+$>=Fs~m?MsGf2{V+?WXLuL4Efc!{fPCQ(P9#X9v09h5bO@6K<22~g2DScvwb<@@|Oe!~*+^-$JMU1u9W@_<Av-1Ww+gg||c6$}@c%2+oM(#xtnPj;cFuHGJ5qB4DMP~!yv4#nTpuOFjO@ckk(7p0U0ADzK1%vLmsh4GlV&Pv>G<ph8DTYGyiim{V{G|z4cAdZG**bD%U-7&!98d1m#U>N}C(szktlDwihSeF{t*+5yoz9i}Wt53Ynn_bv31N?fplJ`hdF$-exD4zw=^p1J!zBkBcY7Y#X=GNw)Rlp1&zy(D0<xIp4lg$i#iVGq6hw#3YK~|Q%Mf=?UEUT^I_FwOh#f|&&PZ(r<bmG3POP<1Bom-E+d0?%8cKC22X06>Z+g_DWC*mA22^TVLU^xYlFg7FN#CV27dS5SvRMozaZ`7oT^HT-CL6IQAb*^b%4t;9>qa7_z___DBl01LNDV2?&F86B59g3B0gErAFEeGpMrW&jU7Bk@mw3adN8Z4*WW!Wd4k^2}j7P_LH{8aUuqHD!1mGy{JKzngMxADNc1<RhH3+@HjIUo1=E<X36Iwwro{s?fg&yLnS5_^X!iXQ;3oL);6xR&1j_EEou`{`bLKnafnEnwrjRVWVS3R|pG^7qP;$Yb?yl6dy3zB8n;<6;l92PP`xZMsYN-M@g_@R;YB4YAoiWZrVi;&(`3z=!_*IHDpdwzyhRvN`!G+I+>iH7h>nMKm;xB#4n{;@F+-(oB@=02XbSMFeyzO{a$h{GJ~Ns>QW(W)sN5}2nW6*xE|YcLMo9AKJynTeLz%l^w@doCTQ=J;3);6A8fQxgo%S)qD<N}ekt;xCHY#G7EcN`e+c95WF14@nP{^PeOg)vGPn3Oca`q=O+KILawJeiSh!l}tq@)RLQW96I8BD8<&@G>vpV6`OZA4~J1ITtBaley_-V2hvvphBVpq;?S_oQIaGo^L!Y;tH$pYvOxoKghWkv;O&sEz{++tLI9YpV#h3Tw&WzE&mA;1K1kx{TBMbY_+$}Cg3Q_yV09oE#|jfpl*Uh>!<Og-bIjvDgrU5dq{Ok3sNAd9*M;rI>C)I0z;)cu7eI^_BVE%(??V%Eq)3X=L?W^0oL;v~Hzw@dgDzy&YN)(eZ;D<5sKHMUj5jar+7AkoNY4Jkkg<3b{{OR)G0K<~G%ci@0vohQzyPIjlBF6%wIFJh+tp+XWShwxi(Ft5xBy86!1cN^lTnErF%=5xh{fxJ>S0Mu2t#*bZLPQ@%J06qq-?AukApD|bq^7Ep&;kU?F6TNS{ziMFr0JwnS1RhEYuhCnwfBTLOz0r&DT<Xm&8S-NR_f+qA-V~D^Xc(UP!d?T!2Q~KB!V6uH{s~v7EXP1Ic4w^>KARiTff7;ZotuK8)h=GlIqmOD#iNv91MirF1gDKJW?Oxpa8yq9IpMk{Xfq24fO}JrRR!LZ5Gtsvu3-3#cxhLTg@y?8C@wzB;`SIZscJErqsjF~)O&o_yigSCtF}&W&g5!bm`c8@Y}}v+K(wRI&Y&ZmeFrp;`uk_^Z^)(ML<>Dln#gGPJJ`B#X-I>qRF@qA>S*$HSiFyFad*?Kh#w`EpZ}qoEnF8JT?QBa3nt3E_&;sUT2+u|CEVk@2=f)JO0wV(yE`7_s`8_BSQ(j(NkW0f<30lxkZgNurWxB$<p@-4=q3jc1s`W3KlO8D*?qpjM*6CxfysYdI?n&<++0>_h56e%efQz-e4ol0gyECrTG=;%xHJC}X4uZ1Ok+$6ymJ$Z7ukdG+oWh8oMO2oxf>=p^EVvD#E9rkl>uK0*vJS*tF|5hfqJ!N&e7rN34(hrxB2XJ|VLS3KpqX&8YlqH%e>Cz*(VBdq6@i<rk%fWj!R?|{6$RHZqB-)Q;;l9D4>om|cnuRw3u)ZTTQKsdxdtXRy8M(4r-Z`_?(L;{Oi!&y*E3G(^<h=|8|WDZ&pfT{ef`x?cyr^9HL(TWpte*>R1f<n4Ok|<%YRK;*2@dX;^m3OM@0a<Dssvx<;s!uE%|BB!uS!yhd>ga;f=(PEj-=o#0GSr$`M$QqI*(N0|axZ4rKy{;9HkBMQ**!Hn9~Kcxxh+SBt4V=bmnB%&efG#(sN!n^KA=fG(1^RO`C-Besq;k5DPgz+Bh|1@YT6;zz@@s|iM8sby3xtlyi&TbW#?v_NZAQ;VAN)ykNFnKF8b?|4)ihWw$x2i*A!lx4e5tNbwkjgt5ank<oNhid;_jyZrXh9Xa-(_Z6B5cY{xb*bnk=uwa1|v#EVcSYaUjKD^RTDD0Mb;iUF%>VMc$Fky0jv=at52RJy}kP{81l62_C6>a6~&8c^?Bls0GdX^ikWTn>FoXg?=%%on~dK>VBiL+aAdS%Q|PFVaRrR>Lmn$_g_<Q1`-<N~72@$^gQ2186!!JiMu_40cPZpkOx=q@-ooo1O!7otb$Tad7}(=YzFlQhlVPX1Q%y83vwVLZgm$Y+sO#NI)1;oQXl$yegB%?~_0WDGug0S&1TP?zMu21y=%B<2M_NRrHS}+<#$Ye#41=MRfNQ<8_!OM<c0yTpY3VWN3mRQa(itn~P*RmK5!S9=uh?jc6?8QfbS%hAKqO)%L2(1h{9*s)&P>5ou+C?F{<;GeUZVGR0Xf7?7Y55N|uHia-O0qS=*W6)ZL>ky-~jb0<X$@~IJy)h1KB6L{geG@*5(*mr$0@O$1OJ6<yIq*8U*)XS~COR6ukFNo`(I|SiS?Q5y?5)V<H+m%kglGZa$q%P2tv9|=*uA*qK8ioES1i+Z6wgA*+D@G(p3-UTSuPQI8DMwPohxd`M4q<lfGJg?96xX&el03x97c5RAUh=>R<;wE1IHdY1UrQFkO)SLgRW=Tbt0eSsXAvMJT9DV#yBPjO1yoQ$v|x1%e8CNGI^0aD6d7uAi^ZuVRFsQ*b`rTuEZT~hg#M)r2o<unu%y6CN2Ov>u7sWthgg%!5m`N8wN@;~<t89z!G|1Ojv+FS5?zsP3R<%I{5W}AL;gs&DC;zvjlC-D`fhcq*x$1ke*m5{OL&qhKQzhTD#<9&H&E#cZI})5N)G}I!7fdB?lolE#F;Hs6qZPN(`xBUlFy|pASP;dw9~mv&}RBr=-tX<&M*SBq^x2!TP_u=c#<yDz;dn_4MnQD5s+-m@zAKohcq;y;mptq8M(^x07M+Y;r)atM-+c0Hz5lxA45ITqoHXEj$E#=g0LkS(44vDAbw7{BaLeSoISb#xIL}@PDAe=4bqiiyROpyI^9WO)qk1~r-UXE+uvS0Y6;C=kXDT&SPO*#Ld&f1f&f~*K@%X;L@iR;UlON=&k0Di86Wn-qnq`Rs5Melo@teG{go(1X0Ce6g6^bP+mu(4w3JxmWI3tH(N!f0#D)fc6w6y=I&lD)h{)~>D}P$aH)fpk1;s!Q<9Qn8F+k!3bj#z+qstTEOU3>DgmjFgLxCL=rzUB}8Gm&Ys&#dam9lm`Y(d1IB56<&QJ*Xb5!b)r{<F@W3QNSya_pxP)>2UcCBLwV6pXHAtO?tQ5;vm7zfAE!sQFIuH;b+1)fLn=YR_dsU@0Aorq4(OcV6kFD&U)0e6h+O=V6CWAqi$RVrUYMz|l78$w~nY+?k8W!NvEplis`T{p6$o&Bk9&3_Aid;UOh7`692Xp==B=Cmb9SZ8c~NleMW_D5-e?qx|mXkA5*+`jw;X)8@<q<f3}z8h4KTx;7a-BIPo5;;5`)lEkf6?$%@<=b1;Nk(lRmwbWZloc<kqWC#OXP%d$72#V)Qq^x;h8-O~b45g!IDzi>V7Hv~P6YE&X`q`mVT%D;VBvrA@7Y#8?1zn@z!loiQt7By=I$Ah0D<1SzxZ;)7=xM^a3gt1@*6VTe3mOA;0R$+N)+DmEL{BjW>xJ}PeMCuyUyfQa*KH>_u$yTDwJns3%%I4Eq6kxP5?E29OnF>KP(%|M%`F=4iJJ_VUeq_1F?d>)2#<+You`i_rX>p3Cd5h=O=bKQMJ1RVyS^lp0?EZV(`?YSq)n`^2woP*ClBGTP?|?lHO#?@+%W`<E}PENOuH&rH<qRS2BiKZ{F61!t7t@_ZJrU5!aFs{nFF-gA~X|7;>tzHwp<uKeM}+QTGPyn4S-g;EF!gXHN`PluW91vlS9!|dp3xh$E(Y68L=c`%Pm6yk4u+vEzvsJ3<WAhqmgtpr)Y&cq4+#&vZ|{!97Bf}sN|hl>R}}eLOvK|=(3CMx)PVNrK;@Ub(3$7U-GMfTV4i10Z_z?>OOgl0LI7lz?9Z_2|b^^jzpO$4Kl!#5WfJ7U<(HSLiVtBq`!9AVcCMgR4(GNmGi8Q5jz1bcy*B}62?kkxZs7W3uRN`Gaa1k>Eyw(t2!r)#<8lStz9bG%1M5PN2?=KpzjD(5vsk_sP@o$V`a%ZuXmb!ZuNw&Qj{M`j*~<xRK>RJTy{Cq890))a^IqfoSLIq!6=_cR_c}DCP?S5YR1?#;VA>}Xu)81g+yySvkWMGN^`r8qTn1E$L3lKrQ}s=pTR(#1;yCEaVekeXGz{=oDaqcC5~zlg;~qz9E)beWQ-WDAk3$)6B+1Tt}lv<LDhSheUjgAUESGV_7k8bHj~Ge6Qq*<&;u!1O%gn5k)h(unpH%E%p;E*R;-jnTQZVHYCLcJfq1%JH7`YlaT#mU3rDT{SSZ0Za=9NSR0`ACXl@Y!(C{4tRw~s}W&sLJHdUZOBih94%fQsdqbpFYklfers)X6_Lsl0S*xr){cIKBO0xN<WXa0lR(zr_JE3Tgevj-`EA*1UAvik_6nyAX9k(okaYL_vVCz9!`H2s&=CE7|rB7iw);*BweyA>Z|?l@CVpdPxf$(|@wFUlO3YcKBZs1$wjooh;8)ImgSZb*7zOfX+#mB-Oaa;E8`?>3vo$;y;;l2m(E$dA?hQnw45G<m>;DmT+2vH(@hq+V1_Ht3PMG<C+$E)ho;%<<A0H5QM{saK#aRtQSUBN7wP@YL+QkBXR;TpYX?xOqU!Lrn$N)8}&MdaUyM*F|}M=Zo1-B376J46X;c*1Zl^E@H4yTE?5{le^XWf+wkxEjIKTmv*TFcSUSxyc`4IIL1miqMo@|&2DF_w|<3;*VC(L_b|H5tuU_lHQi?lCGj$XaRoLug`p^vA{T^+*f}$GWrE%5Bse(La4d*DECVpDXXN#S0k5J^2}tiWODhic*W<q-j4_vI$;8>pc81a{&4z}HJkqL6$3)(i2w_@M{AuZ!a3z{pkI0otd(73Cws=naGgrie;U96&qaK8VkeZi%Rak$HQd_P@WL#y-S3%?cNV`Nx5+__|@=_~kkYpR_c_ej8ZZ0Gu>x_XM(fSgYac>q-zT$NUnZi3Cv7Drhe(-@q-BEYZYf7w%#3Ir4$<Q3S6cNW^Wia2@gVB)T39L4y)MpKuQGq?-q3QKcLkKFhG=_KoA}8GhFAOtYYLzBnWdu;De-@hJS!fF{Vny#rpTP;MlvdW$LSr*LlgfxH{46IrL46X2G<szX($M$3*_^7?HWUhFrY9Xan=}{$rjBsErlE|4cS3unxrOLt4)3cJ@Jz+w7&S!VyC#n0q|TgAwgc!<rT~juBB7DQVKGw$X-H!s^9VscdgvzvSTIgDbU;R)$}nhEe@i-DvALQG(RIP#!migG?lQ@fWDVfU+86ZG!wgqVYl24GOIR5bz`Rx?b0zmv;d2xlUMyU~f?<SMTATl)%_qyWV7CmzJG5;r&!V(_)d*-3PeqLwU5zzJWlNHyjm(`!ElsTAM5F(PY=~4|vqt@^f=b6MFTFm2bwdfrj+%nK4i?S~j!&dco|0OvC71T`%;HR;&JaIPD!p0S5Bh9au1m3{DiLBC!lfnlBvwtmLSA|7RHrBV<Ln~=U0`)kuVO7FljWr5v{WQk^1;t*U<<-!S15=}hW})zM|tc=S5_Pz^4Yurn$#@jZ4+vM>TF(FR#2B>=0evgLD7Ieg!7WqvP-0j(L}?^2cdvxRb-1DCWv2dh-4Kc)`SQoj*(dOJ2yqqE$vl#0g9Q|bg~WMor&V6#4k!|TDU@noQurddqa2UaTs&_#{iWNE(9Q!L&2yC$efbj<7sv|dZLguPqRrVx|%z0vd|$Ie{f;JF47Q<O}S`N-wI5VN2OGvX1EVKxG7w1q07&d6Mm6Op>y}LCKNFbNvO02g1r3wm#jN*y5&GsxfvPxnYF@sybx2_#@Z^B&rngoaXkE5@^QU5i;v`Gg0D7}bw!(J!0Ko~wNQIStx+9Pq+xz8=Uf}_??QnFoloU4s~m@=X>~P8+sU08hwR^?bf>8+l7)B@B*=sll$yB9R4m`Y7bH$JO-~IQaTirnMwX_T19I+W4|_06iUd{~a>I+G!oW*Z_n`ZarM*Q<<Avm(x>=xrGgpu^0x66;RJ{wMJWi>81fv>grEF~%t<J+r<(p8BDb=Fn&T_om>$2sBb2Oh7dTJpQ#>OB6S!X!6A(PyX3V4uan9x?U{dfTdt;&N|C^w_#n7rR5=8ae^QvCz8f-ghV@Q@AA<-we_!cYs;3#&}(7&(rMJcWy8w>Z@^N^u4+T{uC7pEbtmB$A*~Gr52u^QxV*sst8R^mn3~5zrW-ZgECz)y!i!Fw>-Ip0&j&hgQmt?fVjW!JV%>2miOO)`xzbN78?FAwdhj?EELJdp$dURk%`J`#fPcmM)0&g(NgvBni9JsYXJ6i7fmvtxxd}RIU{V*`;vI5wK0*fI6U^iz3mQl>Ula3RH>Q`T|>l)iE@6gj?rd1&kvj3;R;qe%{2nAbTP(_$Gzcssj8JJf|$jUJ|yrXq@D$FH8@2w`_i)#$-s}YkP6!>e7^ipLn(}n?<-E8z_sg><ZaEvAwSlFS1-j6-80F%`GjIX{}0s8?Z74Jip&ZWq2pqc#mA!T-_7RCnxs_g**+$UC3DDRTnH3IR_i`j~-<md3@AmK1+(Y%^YLtrK+$LL-D10i$ycYvZb4sGPA6$LZe$L<ajL!E3QR>QTyySn;K1zWpx(%WaXkYXK!SR%aZI>E*>yUVw|Sk^S@-F8cBZ6&p-_m;HgG<RE^ru$+JKmV0pu==bdPrasHL+a^P<R%5#e&D9>5;-fnXNXl(JSO3zvp=94W+tH{hHuo$Ny&Rt)X8#%e+0|xjyL6lRqH~mW+Lilp2jG#80>vX5Q`B*L<sZN+o<Rn?;!J@xeR*modD?p?x$E#=zJzmH2x?%{7{3=gY7|m;x=0cL4C|KxtMa(YaaciPg5WuF&-9pW?wdTGGLq89Mn#r1?&2|F|a&l#C29^==I3^t?=`Wj}K;^R0ic>NcqYvvDd4Cd8Ft6f7adJD==aVaM#xsZ|k`r~Jy-EUd-f%wG*{sUo=303KmkYhJscX&EiC;qdit_A`FD~T&SrciCk}JB03OgX?%kyhPszFP{jHHT2acwDrvSF*8@pXKyyhJZ%5z<Q8&a#8SdX`b5U~o><Y+H7ucu5;-dZ&78rY4CN6lEZhz@M@=T#LTYP%dR+dZHUj*^CxFsH!qL1Chi_1B3#^-!aSb5a>=rIDmcnJILdVM9EV1Ir)8^NQz@A95nDPKm&^g*==i8Q{9c%y8uH%lmje**xR{rt0mWlN`(S2k{dtb(v=rQ=#t@0^NYrbd>Q|$P&d7vYwQ$0O9-RUR3>Z(lI0g#RjQ=6rq?xXer&2DE5$##iD*tpr7u$rR?;iU#5Z~N3d?hI;HP+1&5T7I%mi|TW2LxXNa1%MPjuoSprRKPc!;--Gh}uzBT>?Fbc{=(P2-%`mF*ii7Bz*$BnnHU(i!YXla^K1Vq2eNtNdX6%M2D-c`w+klU-c;hfu4gu_`~ZU1}cz+VFCw_`L_>4x(d3-AG4iSca#f?`4F~Se=ZBNHOef%myQYBLDq_X66*^V*OU^bSIm1trXfJ?oZ4TcFYgQ5M>~oKsIB}7$qOJ81P;#ukN{xE+#;VqN99$dZzO_g((1?z1sZ07-Pi=fHZmSyZgJ3#{H!A)x6OOM-RZDv%tl<Z8J5!;Em4oR}aW*zG0H}1^&-nlI1(gWl#<VxP2zSu6!lv3az&jmpfLv$%xKAQ;gDKgFR)MX0NcTH1VveulpTb?rG%%3;meJDG0PI6#j0VasJe1($YCTayI`1P`HkW')).decode('utf-8'))
_LEGACY_ACTIONS_6C8S_3Q = json.loads(zlib.decompress(base64.b85decode('c-rk<%Whm*a{L#rxlld$kaujU#uA3z6ewwkaf4_y;4uss<3-y$!~brXDpuXPCo(c3&nc1`a91jp?mh3585tS*>;Ihm+wXt;{cnGq{L`-|KYjgp{r<O4SD(Lpz1^JLpPu~t@BjI)|Ml%J-#-5R_dovQZ~y)6^RFi#K0f?a`|#7(zy5ah%g3Luu1`)+-rd}toGzQMKY!S4K284cd9!){?d!V_o2xG;rx&xYf8Jc*{Bm--*!}#|?alkI@4oE+$NByJ|DH}e_UYrhKY#tSf74>pw_i{0HlM#dwDp&p+b<s;KJC7meK;J5&ztM({aaV_w>~~@@+#1f>1+3&=2L+hFne7%d$5PQmORYK;-IhFUy*lxy1sh1iN+K4=kY&)x6Rs3-n#uS)A4ND@$lU*`^9k3*X>LNKTAh=b2WeeetBGd+1$<-(fqr^)dQFAa=wT@-+Y}fqIPlq>Hl}e!8fzsv8ika=Wu{$qqOheyQ}TeeEiY(ojK{cHJAJ0YG3*|3e#Vu(*^b)njEkbniWjmvK@OcW|QG)W~}{<K4aT)r$cw}-1*Mi4`Dk^!Ma=sha1=o;nB*^mV+*6Ba055eDWS!s*k1oO+Jrc2)8E;n4@go^g-OcWB1|P+4~uN@CI%_?mZ9Q{*q4m*yqy;AJT!x|2uip(C4Nfp5d{xTV)kklgVLfTp(kfIzL;T?fc{{nA;=dr;QmgrUmbAuCF)mzWny5&Fz<u*B}4o@Jtvqc;%NEOQigcBhA6%tvzW^xQBL*$n3|#RepAFSb#5j{TuT;@8i1e-KO?mr%eLPyT*K+7~x>yR{RWLjKDpCd$nEKmYK}^Fzs#D$8-RJV{aIw%vFJ(vInxUK%deFGLJyCA3OZfxXDEaDjrnH_Ek0z_09A7C!S89>#G1y>EoccY&Z|VxZgjrH3sv|-vTGZw#?gSJuWp>32yephV|>y#y?HI_kj(yRzdE%VG!F2?csa~qc3K#__tH<_68v}(s9VHTIrCi*blq6P7W;p?iAbJ(>ZGh5whNOC(!$~%h;e7ZDm-v9TP&4j?<L2-!Pk~<v~n_f;~nT{T}s8u~C9vC4-SehR(q|hqC^0fUDQXzCQK`e5?*&jWBiM$U6+-r;yW`4WI-e`S#t72g}?!4PPmGjRv057l6zxq6!G&p;Au!NmRYpk!2TrFg72qZvPT>Y}}1+pao(S8x7U2FU296jztf}pdFkx2Du{>bO9-R(03jC{#I|$kx{i9l<7$2a0tLx4wl`1jXn;_KH@14`u>UNs+qoTVxa37%$%b^?=$d*N`$$6a7&YVH5*<Z+Y1t%WtP+Scb5;_y)~xJF!6EWm{waO=JS`^tGk~zx3_-<mP`p@N;?$39nvtD!{H{@z@QO}`y-%HPY6O;cc)=S&Y`I4T^b_`Rp9AZCf3wCnPN>I!o)$9uJy6|u)EUp$7wj*<BdGow3z%lFx1)Qoy<p|xC)|vTVH=~X4N7@pT0M=5@Ky{IYM|7xZ2L+Dlp;E=y<Pzt>+4pojcgkS*KmIx-T4`LcA=o5s0CJj#tH*roMUx##F8qhE_~2!MmHA>xZ-$Xj*OmxSgPH=i~d6vaPrG=W%b1ucf0?GY1((Vi0F#KGe~zARD}gvtqC1-9(6>9E>Gj2<#t_sWjS9C`}RbL$vsqdS6R0R2Mx=_g(tfQEl`yMam@fw)s@XJ2z2&gx5p>o6f_rUK0_@IN?kkCKhz>0;BWobfcqhep*zv0UKuYNgot&>a+mzJawkV_yRu3%)C986?JZA%PvY0yTjPnI=8%F#;R?OiIgeL(UrqnGhT%Uh@k9UkZl^Y0bIZAI@8e##h=awDEPX)$%7wqNFg{cR?cL@jNLpv+p`d)dz)tD#Z)f^QeXhtF0y2xePI5tl8kCb9Nz_>NJev#U)p0mupLAbO!a(nmcweVK=>az4d9SnewxH1TU0Q=Wr+{L_FKDNM%iX<$a%i9;D>yqnLV-V1uy_YTf-XXw#)3fVgEdnhl1RsgH7NWWZ_npw(?3g8VJTMo^whIz?c`#HfNjz*cuFf<&Xsfs&hU4;b$rRkB?MZQo*Y!k3dfNeuiA|F-Y#h_CfktfN!z4VF~0+35Bgq0DoqA9)<J`aJ%*yY=d(Bc30LE4USBIVAK(3j>;ctLLr=(%I#OoQmW>87_I@hE_W0BXz>30@%qo(l^6L*?Jx68zaEf$yQROsZy`bl60$yx*R0(s28+&y6*zd?<Me@-?N%b#@`AXHm7imVp@h`aW6Dv1BYSVAPmSRQC~2*E4vkb2dWZ`%JxX%YshAsOjpZr_Ok1}NNnnX<5g(p?&Ka2N7By<maU>1Y%CEIXX))&7(q`Eu!}m&Tq30A_4`&AN)V>P;R&!1y$G7mInsA=EMxk#us&!-=(Ao1mTD?7N#e)m0b*#n?51bYXt)vr4oWjA&tbIWX>C9bm&r|Mj7B%3>cOUz7*!CwjIm~-n)yo@$Nd4&cNU3K10l|Qu4W)C-uCCyRucI@C{7%!TI);M|43Eq|Cj?zDwyC@8sl&?b7%@jO&z8(>ewayw?M&I1wUP&W(lD7P->zcluw<OJy8=W}&|3b^2wUT|1X2qnsE2wGy*R|K#dRjt^oltg1qJY-{;(F=ZSe`(RmWYHQ%2DC*VI9b2oYeNHQq632y#mkfsxkst+4r$7fbPb65u479CbnzIC%h4f(94t&R7$hKeyMH)`k`;ZMSxEfp{j+#aRX_Or5c!GLEPUNLL|V1LRi}!eNR9T@X=w3co{{%p-m1oW>_;RsH0734;hsv6s-S*TH(-p2xu`WbeQ11A~evFD8KAwv7hRh2XX4^P+C(Hm?#(ms`mqB9&0m#eJXTaWP8^nExu<p$4y)Tv3lJPW`9~vklN7Y4nK)?$AR9%3<SInbce42Na@qtz*n!Z`xW0PSqxpJGaxfh+)@bByADDHfwdXFjEZot>N`1*{Y1@i^MX?Y6}2$eZI9Yak72zP{T|^(A>qFdjkYvTy3&00K%~U3WnctTQA#SMO=YGNh_rRa!3$k0L4a8nvi=YEF^GrZQ{mA^TM#adehrAg@W5LJO*;C9$9bGI*sF2*TG|x&b4RJGzLvtNpn~UbC1NLX$RWgVsfR>Lw0fjKY$Jc7aHvD<KgTH$%xV%fFHH->CASA;{uYG<sLg5Pq%7mWPo&kZjok@Z9!2%<I-`|o{_?6u^Fk&jEx^&5a}LaZH`l|^EK4hp)S-|t@5EY(&a#)oit%k)5^g+@ses{^nv1CIbnf=FwdL3(DF5PZ#nhHL(g##Wdep}IhUM<QL}CoG6{mK`_fSjLD4(NWF9_Gy)roaED2b18GTu){&jG+>epqs_G|ezjBk_;Jj*dG72}|VGTJ6n)FdC#*vOc%B`d@O;DPKqkd3QGnP!)EPX<;r2pfSB-@hWlqvy0}(*aYCry+oDA^$xUt?XJglmtQ3W?tfwVb(F-#34{7mrhtX-X@Vf;-;~&EON2aI5$J;paw+oMVnz(kQu`kSENjqkdK-Boc2IbSP34&rLd}Q1aX&HtBD3#j*O7rS@EpaHP)7Iw5C|+^1Ldm>CTeTno0{Z*z3e0-Eq}74e4V;Zr@>KrKC@%&6WFSXj)UzJL5z&5!ITAAcYRyMuFO==x^2Y;wi~1PT4jax7lMf{n8R6ftTHv)An2@IL+~a7{GE+1Ee7mTr)xSjFdVnMg(6Jg^4$2f=Yn^L!2@Y-?!uj+W9Zii~1Fodqtes0@9HX5C-LBojhJx)Rkn)5^9}IIj($fBq}Kt*3+zvatalj_%!i`(I-+bua9YOm0QlWA9ORCY<h8I*yd<C51m;(jL6mF^a}Z(2Em0%2v3|H(icb;>GFv%{Ui=qp1Or#;<VWkS~Jx!5(2Gk$Q5@)Tn1}H&7M-KPNTU2yKas#3kP-;MI{Ill6NV&eiTkj`*wL>I>9*Jl`WZ2-cxAqX)j4Z^^j4|!lYZDfyfF|s2G;cgekPIRm{++QNU-GV<RZ<Nog#wpgO#MaAwRqHKd6|>jbr))zs`BLEGp5p7wl+nHreWibXAc2xQ>NI2!HI2szYJUO^0J6aa!&T%gK8aCISib==j23uLp&)QeiY5?rlCY#?>VDwENPDFGXW`>E_|GY6e1?9t$QI$~<ddqvDkma1vV9EAEFhlp$LBQVK9>Qia>a&{_^94^!xS1RG;kwdw*m*H5d(h8dLr$j!b;1qEsff+9$8*VkO+>!#<m6oi=qTYQq<0YQ9maSEi{insWIK0S+3ydezCBYllkjPd$A;%a^(~6IwqX8K0{IswM9t*`eqb@me%9;<%4JpX-iN2xJo*tDz?);4*%NGj*yoBKab8gjT$lLy;)e>e7@o`Jye)oLeP95BR4bPVq5$CB0vXL;>*Tv8+kc@Af^{%R*%xVy#;3Is)J1=<MUkRa$*<bWc_3{ke`UV8IpazaUS~4|(vhs_ee0?OlXw;f)<RizrMfrB$SaI>Q`=0y@%~mayCDTmQTtlNa8H!S2uo@2Mp%O@MeSXn%HW>0=RW#VAvM;H;TP*<6Q|1{MU0@SZ`f_D@84$_+)ixMgW4i)m@NkA{aD7OXNxFw)VuYtQPvjFQ_rfZOL9GckiGu!0HUCT$>;NsO^;k-RXF>QlA0RLL!Q+rxc6_|sGnimCD5!TX2ldu%Oz-UZe4Dt4aOm7T5xW=0@YqS8c|3!XsuBH{D9@Xf<8og^rSC#^bo!;mX*Oz!)QP>V*(%VgVLvN6e!d0n5r09f_!tghsvTWa)N(}SH&C1Y+=woE;!Ye-x(H>1F7GQ9jljXBPw8~IrkVPV-z`lm(=msUyIAQy;YHTPxmJdV7*h(%Hb<mQ=xP=edZf{LsrZ@bYegn|Nh8h5DP$B>8qeh<h$K3Mdt!u0-C}e2z8x_E7xzY7oO;L-$vWwX%mxPQW>LyTRTE3-?jS_?9<{)=wWKuLEaBsGbYr8gDUBf99=Td!9Ofqq9WZ)wiF_+P^93|C;l*c@GO{NtLxF^>a7HfCQshYqt5U}<O42HD4&HzKXOeOJUEXA(S$3|U3x)1AZ<=@eRLaq32o<X-D0~y-hon-~z4Luql^5AJWH-~|g`gxqm1UicUakyTLW@iw1^rn@`|$ZIasm!t`QXGY)`1td1iH41x4|A<zElc9jVV8jG{8#|c9k6~&;>GhLl^X`5Mj&Yf{H9;MVX_QSRiK-6hX2vbOYqtiG}~VgD%x<0YRF-ms*h4Y@l@JaZ`s_52EH>)5K0*fdBw|;rDTK#3E*BXi^GDrC4sj{+&uRpUWA|Oo8~koM$bBs}tV0aYvEY<2VRiGf@P-vX(SeGLAb{^cN(pxX@l8;s3l^rM*;QZ8)^ah<5~jG?_tb-dK)?E68BSua_k?0Ln~Ci1?d93i$km{5!=_GguEK8v-7cq$Eiq$u~bTodkCsrpP1?H%vmKPx`iyLOxK=aDWQiq+<3Qul|7u=+BsD9n&zl^>q}E#(}d)D!f-}(f6D^Qp$C7=O%GUhDkqJ$~6e`4;RTR=$<_`XLM$ERdrv0;49S<SSEKxgv_WOkjOKt@wMf>k@*dF6FizgV5a2NYoPG@exNcOQqjqf5uMbaO9>6i8^4o$bAi`AIATtHDm6njr`yX><_jZ8&KzWF#wNkeIFOsd0jNu7PL!N-?DOzcv<ZCGwYJ#=NT8cF6`#pGHKP*Xxue*xrqORC$p=x1x=t4XYy_H4qzNL57@3wPCGUx)K~eyHgKJy^)4<WEQjr#Az%N409%H&hpe50eY&%G6TojL7F}_TAkA^`NpCRHts%axk2RgdY$ULQ$3o#dC|C5KcFqChhuGfTMPmy7CYVg`EJY=J+-t2H`<MOkmDt9x;G*`e?CB`?7sg|r~o>nDAtpYPE)=Mxhy3-?JUE4~LfL`&Y$zD3BCpq0%mn^O6ABSA=%+d@`29WiuH_)gUZ@xy!7+<fTmgIydvjF}*uQo()9Zz2lP05I&r>|Up!mr6VLT=m88WA}x2HA+wDaNVQX<}xFb|Tv^X=%nHB&gNc5S8N;Ewx&tSTFZMb^j7)s7vUFqIP`TAdf|)O8KjRRRTc<oLV&~8QhsqR6H6P_LWmEc)69$XQNhuKkgoZ#gz=#em^pml?A!-7`hmiXVjUr7jMge3sIzYU~4cbx${>Bl9mSIY8S4gIXWsMA(U(%U2a6%Sz4J4iI*`rIh@B!Qj*jplDy_@o1_OVxkK1rt9<G-5*m?xr1F|Q^K@o_KYV38Q}fG!k0j1HqA%jzuXaVr2hwH*X)lmLsV}LTWX=J^)Dp3#pwmULW6xd`O>4irUIq@8Xz3_KEvG=naGq9%Zs@ricOnRPcGSWdj#f=>sB}`f4&yr<hw{20YW)|h{kCf-f?8yjmX4>Xr)ED(ODh2o%ZWNF{Z(!JB7E5rR+N({@qo)KOsUz?5<4_}Mi2|}{ceSFrJ|qV2q&yLOJPysTQOFHaSw2vkd5eO7XZ=}A>v~X#<&n3eRJPe_Cti~D3hu(;urT$VF{72q*XJCsek@Dam$D3IoN8I4Aq2Jn%*9Iu1-)!y|XoU)@7=O1@ctnPjp63YE>FKK*^Q_>-|X5M;(4Tvs6fH`MmH%Tff3nhYAK0vmw(=7plOYG8u};7O^3)Q(J!o$O*T<A<G63a^y&XqHxF1CXpW$XB_6Vq6(`KcB>nZ0-7`QE9%QcVl^W>H!5TB0J}Oc*Wb`w!iY)r{KZ{ge2!rP-K-1B%11-4(mOiM7XuwAh%7S$$W&1_T*|=}YrMonSE$uOeU4fAwO#S0{5utN8(SV0m4eNxR7h7jDr`|isYFQ+Fzpz7>enM8*b`U;0nO_0U+JV1<Fi{IsC5~u$y$`qQYuVI2_-Fz6D(EaA&RV;+=gr~Kco}ijz?PjC5AiPIs=qe*MfP6En{f`c#A3UqLj^m(=^FB%a)|Tv+yQ9$(e3Z=Nwv;a#d}84NpXc^Hq(`l!Tt(`mI-r7DdFmHIKm2g^9Epn7RKdJtdM1Q)PG)xGIb0lo)8tvkzTbLW0Rk85!6iDpHec!fB;MGYMsO%4pTBxI|VaN*1VT=^fIzSRledD%6}7l!_?Xj08A#?sE#KNMHyneZ~kvaCZq&$9#z$pZ0^Y<DRh&@Q+_BYl0vr7|L@O%TihT&s{Go^~}X+(Glup)j=I<d3987RuaKT*wILpA0i}GP|mE#EE$qYd5w%(fzQ_SoZRDc*SNc9!ot+FI9UQ%P}DBw_E3x1&03%5a_3pDimqs2!sp7XfkgnuNTV?)mI2EqOT=Tdj4eY4Pne&xAw}RwAqmkI%{h8v7ELlSBr_<R7w*GFpF0x$&V~Lk+6oqFZo^C1D;BfA<#d@PadGm+N4+_`QCcKfT%B63>(Jsp%;jwDfjG;UpVx&Xr*mi4-j;M$sLRR169-g&zz12!ySj@#(7dAui7;fvDOr+@UEk`}XawLO(n>Z`+YYbo5Jn%fKNW0@vlWflNpSqy$yM6PLrTmWi5GGX^O!n`19)ayS0q*Q!6mZvKENd>bu5iWJ<87Y%~CHqHa%BYXRSulIoD`PO{XKnW^06tttqo;Js>SEx{MbK!@+_L16bMg&uJ#zS>E@t+@v14Zn(`@JlX)@?&2si(IqenGi8ZjQpGC^wxW?j1z=OeJWFCiwK^}LKr@Md7lB8Of3170P{JbSgHd(fjU>0Gnf=Tuy^DkUMUs(XN*2oyab5xAO2Bvon*Sp#VbsE0UOhQlnB`ZZBe7L%o(^G9cGHvGsn@~a`lbwDO}GIbiAF4krI*Xo>TwZY+RzcsJUzYepQaQgf!4Mh&kT|fYM8m2<DQ2&wS9|fT$08#o;`)*5^EGmuyvo3u~o+^1w|SEOIURmwdM8g_`JRGLkC*I)uUr5P>e2^EN-VrIt3Y&GT%}~yyj1np%ulrmpOW|eW<urvrJ=J1e^&{5LV}7w+DB(DlK(4FKy3wX_AhfL>KMD#<R($X`o>JnIznu3i|_0vPv8qTL1|_F{`0f?m_0<Zl%*2lbcu{D*BN3cGbPi7em3Z#RW~V3nyfG7M#(m*Q~o3=+P9Th$2Ad;*Czi<EH$boPl>odwuTJnQx^zq+~t4f>$-;uo<m^vQi!%S7XvWL%DiG!&^__OSmk81*+6~G71*5@MXgAsFC#7JzOb({!n`d=h@^_(^o{S#+_n#B`<j;&5=xgjPFjBA+cDMf|fFQEYCFoLuiy!Q|6DX%?$2IZZ4LnGfvb*Uyc-%X(Z~V7Thw6DO0kdXR)^LEiYE0Kb@s$ade2HF2|Wy(ic`FP7~G2<RFcyyew@wu(m|~FxL4_;it-G!@{ixirANoQ6o}Q36xNst4Z6+oK(^*S{O9&UI-!$R_sy9I%NC7Fd2x7vl5~FkDJk&SIa(y^mdUpXa|pXDjIFaI%+8NNU%Dg7$T3v9u8KDnuO&6JEoaR{wrc`VnirYD;=SRrS4p4F2nUOiymYuHPmhw>Wm(BLrWrdk|JBI%@);vsFzZ_cnUajX%O?43kqGnbQQ24q?F=$y_DjMt)N)K7}fYM;7A?0aAIfbYdYq#w%5qYi=COT(ut~9K&0&Rk%_F-*%D<_%;97;-*l`$?PHv2q={WSa$lIkl{lY@M9v5hypqGg6$e8JlG@3@wPN7d1<g(}k}R1SGT|yBBq|X(K>L}Ku1npUlzA8g3Z%rsG=;jo?^vY)t>7EA)dOfnM@ZHsID{_ppiNu@9amXPT__#4VhN1<Pa1ni-i%QmNH-#*5cvq7iqC+FBcAsyAvzy9ZBB?lr6f7U$Bq}2S8kKwFQBYo1eO*`2`)O5*E#mlP{F8%U9>c%U~Ga5D5@1fREQvUiS&t-R>?aeEIEy8h8n0)G8x$;pPuuftWxo$Vl+yLL&qDxdK!JPX(P(e1;DpoWR*a#w0oKKCAwS!;G(2^1<F1$pG%S!t0aDDd4eK|Cb@Q>E+w$e=(>|-#@xuNo<A<rw-V}O2bVA>v_|KX==QL6m*GVsIljQ0ViUJ?%CU=^L@LHPKeAXOSNBF5U#^hAxG>p3i`A`zc7vcL*Ic34o^GgRf}^R2e_2^3$TlMhFq13;A)#)ZpHGe|OQ8|m<5IrRq~Gc}1?W?2c1)?sAR7g>!TOWxp`IlYIoi1Ep2Qc@SFY7GO?1;Tc{*y9dqlZ6C4O#&CG8X|m9u%?bj+v3)(zli;eI1HY>jLj&f(I^PoSJDFaj;oN67re)H8?u9zeQ5j5#c_EP;MLfAdY3Rv`642=Y5QVEDDpA18Kl-NC?#(LuH0Oq3wv!E%#}u!zN@gf7@X!-@ic8VIi*Cq;p^ECNIW*9<m`J0&5|6~R|3%P0|}-I6Bu!9JSk1o%uX@cp=?qq?vI)V$t2bJpdor+pctRwkywd%El>f!JI5ut*28>mUla++|;?Ws02`KM-po@&LwhvM_rYQevG&3#jPGzSOK}<w&mSg|{b^W`OD#X)e2;^KVlZ|FSQ`&Da_!p{2EgWuQu6k}P|0CL*Uz(K0I2P=J;$O7Z_zZ4TrRWi{lb)Ejo7MJYimV}(gX09YaCph{*#!;)zA73Uwh@JRDbtlWo;6CqAd52FOBFzS;q5@tT0=7~vpr+lzw=}#oiLf8O#eKd@K2t`P|M)tg#)us<)M0TA+rnQE4k037=97%Y23xO1sItFR>X11XaZWJ>&<*4I=!1=z`IJ@UWz?#xe3=k&EWol81BqWe;9x+v~61`J$DK+(a7T`^Y-Ij+!iy>&VD$W^~s#<6k{YqW&L~v9y_Lzqf<9G!+<#G_}bFQn)FU~MP^uyF?j9|7GTfec8%%+fIv@(rGwWo@zR6_Kq*EN)HRZi(YxJ8$eHJ~W1C$$!HvFag|FkqMYAdJQ>)oO{1E1#E1dZ?CiUuGTH{)q#Yls*C7a*;SY%e{p|QY2Sn?n{g_O9l;GCG3>L1uz)MCRVB#VTCBRAd??zyPE)PrpL6hOhHw^Ll8em)wB4CZidm=VM7rex!M<=jacd97{VPH5jGdhX1l(V*Q+@ccZxW<x(v;1(2D_{1zrYaC2WgEdLgE1(76%IXcEYYqBo;h;$ZtWIaX|h+1+Lyp&hc(vpQzENOrg-R8k`7<&n08CLWX(_n<88S}tr^*PUSn<RoMA@458Qh|7Z(^rW~G7w3;VuN3GrJ3+P&5g)tLYP1leWW@Fu!QwgKb+>}*i%2USkrNkhyP%TxIM!<m@)CYeuUF&{l$oKP!zK`B@I7@_;9FD!65RE*6aCTZ^>qk|$EbZx{Jd1CPcX1c?Ib~IDhk0C0+AyYCf}l#vw*K$buoqVaxax_ZF-?;NrRnIQO+@wQE0OEXd;iadY3HZSgKT~8-hYzC?YEEm7~)mSZWfZMzV2Y)S6Bsoo+WnrBJ<^a~Z@Xr3>~LPDu*g$R)>|R2XZDnaXbF?{x|1SIQlMx<?R6BENU)0*XYAxPsCF5hMOtd=Of#+dfZgn(_gRXl+i1w`{Kd%4J}mtXWnKt=Crdw`vZX(7?|{;ND(RJ(!SO%hOz3=P{9DCE9<c_v$DBNdQMFsGnAI%uBW8o{4EDZbU>GfP^F!Q0HD#QGWK)zV0S`HB$B?x5jB5M=>Tw4Y-~qmODujBi0fm-A4;<Z$2riC81U3QRtUWR4FL{YNlBz)55EdaBU>lnU@Sq3RnY)`hqqsFtY^24_HB7UuiW^nm%iJyxaB7Uregc+*xI-LK(NUBdVwtR*#3!I{1kY5E&2Sm1FyAKve9Kd%C`g!I<HF-5u@w(#((FC?8l5D_rM?k{4Ukw6q+U_t$>--s+5Rb^m|=P4oZ')).decode('utf-8'))
_LEGACY_ACTIONS_6C12S_4Q_FIRST_YARN = json.loads(zlib.decompress(base64.b85decode('c-rk<%Wfn|a{L#bd0;(QBz5C-*J>Ke88%3^3ade3Fo0GNAgm4}-Gu#j^^*0-%CImu^N3`#N4yn^#msnzyScgfFaLY?@4x;2x4-^=_D{c@{qW_}-N#=)-#$Kld03xq&(HqjxBvRL|Ni=yuOI*R+wcGR*Z=wY`IoaFKRy3d`|!h;zx;ap^QWI~@6OK8KHP84&gaF~k3X*0p9g<<T(3WV{d)7``u6GU{A%>|PwTt;pU=)`ho66bxc~U&!_)CUR@?30&xalR{OQA=zkEKvX*THFFK3(e<I{6nf4+Zs`tkYG;j7Vy(}8$g-`ySGx){H8|G2@cKtqPFJ$@Qb1!}<Pb=BE}Jv_AJc}`|0eck<vyzBGb?T2-3JW+r4{{Y@LYBzc7?q7!ES+wK%yPuDX;iRv-nX3FO9O3ot`2EM_ar?A>7%!sncc-fdF5UTf5k202884!8asKHaJLBY=QSaDPmV<LTz@t$*_V2^(ZfWj+^s+MtUAN})I9%mR_oFcURXAN>|DnkNJE2&?<So0g2V*uEj$+2j-{>>88+ST%C(j-4yyFm-(^OfPGvROpo1uEN^0Vcn3);w{LnofReM|MRl)s7R5e(t(gaLCD&6_@mhj$!4d_8*~(Fbqfj^p0);N36jr1yP3o$xLl*#Ga~O<kWGe)tBD9o;I6iZvM=rp5)*=c(hf)!DwU-h#0`LVjA95q(<l;r{M!{o(1?Kdm30KHYu#*V8kh)8M6FVl0vLJ0_Zg{jEJ{PjwF+9FftFD_8mD*02EI^!hjEcihKi-n$L$zebw`n0JNwI55J&!p-;@z!-sh0{3dSv@J84_hH!EsE^?Q0>|DkNSUhwKSd8@V}U+}4`d#JXg@aiqxB{y9jN-CO17`Efv9gD&p+{W+FV}+cnTi}y=B9B0LK0Ck)<&hZ~hWEA+}}QKI?IzsY-COS2nEQpVt3r^1Tmis3iulXH7-{0+K~jgI#QGR~$ogDz|fJ9VD*7$Pj3R>ZFUIi-CY}#_FY!yc-#~emt(*Mg_dgc{DW^z*}nbAKnO>4UsZW$nbEhEq*iwr~#a20e}SOq9fAffQGBI>&ZXH(*8Ke+52PPA8TS#b?e2B)q`MqB`P0ST9=tKGvkY!;7HQtGr*9x=wW1cWegM#QgYf)LgKAnD7(`uWAo$f!@txz)(RL6x}&@LVh9=y)uAuRAsUWF3qPP7oHBrTU;;FuAbijd9ea7J>C6Co<S;11k&3Yl0FIn2yW<+&56UrIDG&PbiRe-pzHe+_S<&G=!Ghjg;0>AZaQWbp<MeJIyg#-T;kk5dq|4_$e?sHC)<K`D5w$aw9-kg=H$SW&9{vJ=bSZAcF0pFEmA6|^NE~B2ZAq8l(ndd$eIrVbpM~RQ7=~l`svS}?q8Kdbw3*6i8rmB}h|0r^crdGJeH=a<F6sDj8VtL?V+Y$1b5T1+9-I7)@dzYWL9O4`*DuY?+IZ;GOG7ge&+^_P{4;?%?L01n^G(K%dyg}HtEkn=wrLtHmBlu3d{WHd<W*lc;=}!uXOf`vRq>Ch`yJscnR|r+5QAHAbANyLoTdVesNElTGxYU<{3t{MkG{ASu1nL0&f%mMnKz7_(1|TF9@N?eARDp!$&rUV2b}>!2W0*9zU6zzz)0dt=CYMgp{4+;?s&8@jmo%h2wX{EYVj!zKduvDMbJcmVje$bz&Zr_4fsdEVr<?D#t4*ij!v}k*^t=_Y^>2Ib7a8zQzE<L*mI6Y1zd}fl{%F%w94e5+Cej5MJ;jIY#AF@Fh|t3xTc{LMK;rMxfGP0LvOsp0A+Z(k+B@FvYZ(Z0t(H^%GZ&(#$f{W1OeW8oc4R3&WM)NbxAH_L{Ayzyn2=*$^)kBou~E6n8GZcm_sxQwx={ERM^qF%a3t9i#=spr{#OgV?&&awotoeZsXRmdtNsdtJiI%A;zr?Heq=Lkb57rX!N&ahAXoP!Vf725%jm(sbq)A9A~=e8+By5hlifj7;afA!y%cN^x=tyS)+3}cInQG7p*r4N96=10*c3{7Cra2h8%&#>C2g5iS<s<icmkGA<mr4Z&5QS>n=(=Pw$2}BQr;uX2|<1Zz%XP%L7RyPf+9N@W&R4UXDd~*vCWMx8?}!e=CGtEi;?bM6ubhe>i3mrGn-jq6iMf!#wWq>z_W|{du>*YF<+L%Q%&;1q|P9-<S8z=JBpvkX?C&A%Y_Q5=}5zSvH2iJsy2t2|pLCL$I#}?HF%&cu^%GvvW$dCvYwwxtl|)r@%x?il^tV<;L<blf#1~riGpdIle+P5$v?k%`C|o1C6l23Z<>Y$7ma_4(1&gq-=Kqg|qSj$uMe3ZUrud2&2?pf|w}lfzGAP%)C9vIn2QwV9dZgCO%}7yBnOcE{!AGpiTu(;ucDK;4q+VQ2LP*jG#BG<uP1ok~E$%V$hhdFSR0aioru%h(KYkN6j!4zbsx5a@#C_UHM={=6ZYD-;6W0Qh4CG5TWmRFKhQX`JJW*HHAwKERe_}C+u4<s;N8Zah$F&XsG?{LJ6t^026<(p((qyR<2)5)+BT4%T>(17d^(S6MA%A;};It60aq26#`F(NDj3{j|pjQS;Zj)qSf%mJRO8IyHJT;VJ+Sgx1ilN+~|^xiM6;zr&;+2kDpc^S>oGxh6pf?8h{Fo1x1k?$*s*~JDa{V!$*<q3E0PEB_-Wz0K6CWSw#wF3~Uesy8&G(7cO&QB`cNLt1RM;K-8H*-(|w%MtrEzex-0#&ZJK^Od3y?dY)-OFE>DmH@-%*@+8l0j$F$y0#j~k-!8XF-a@P1=wLO5`@i7^eaq-U+J}y*Io~eiPKYry_sVkAMB1JVL0Hjuxs}Wa%*3Uxwr%pnN!JTlW0O!u!8&OZjP$(8HUcQ%yRwB4ME=l2I!agLS24J%nuF9_N2kPVGzu$CnJdAqevUKNMH1t-QzV5+p<LYR+QJMm;G3<!or6qPw#HB-n5?!6Ag|B27H@_f0u2P)ENt!U!of$#!NQgW$UZrK1;hBbt=DC&qh2iHYHLt>DMrFqFzc=w;y^uH8Mp<R7dCldUN~BjZ8}LBLE?04A(nd}&<$%U+RN+Yzq7p~*(kh~G)LBtK$XGlxKD9O%9+q#c8&qBKpPa4LQA@snaay;CUcHrtclb^I39qyEq8U*n6y=biO5!iU~;wOV$1_svt3lBgP~PoC&t0SuG}h+`6@{7089+E7%m2d{s&7=uH&PjH~|9qv=?<kL(i)-6OF&isn`$0#qe-ElCx>m-xr~iPU+Y4MK<MugEBf6b2=jPspuDR$zaR%o}7qrWcgEy`V{y_OXf1DdTYU9ED|c`f)7V3qfR8VESeKQyyuqo=&+}HeQ0IbsY2uovIx782vO4o#wiGtMH~wf2K>m%6@I<VphO|E=O&+q<nks82Jo*KKU<}{tQ({zOUh9xjYA3l64sDhD=}7pspiF=f87NkS!|5lR#!d>)=6Dd-aAiNj&UgdZmV-roF^i{H|sYIocnSM_zFq1JUj7~IxtY=qygqW(}c%z8;P#K%z$Tt(O(OIS$zg3x~()Av!$gP6EwV?v}32K4UtBYbW>54gD6?UTrKRzOAEzmWi)AP^d(`OP5rK`k%U6L$rufe-(l6utRdGb3x3ojBq~HO>3|6a-%KLv^g$4dw(*>2?B=qHkWPT<^n`6(fQKdfPX*urBqs<}7Fg79POh7!T@S~$o;;H$;4rW|ZamDEsY%5NNDFn_qE7m!M4fbX%%D0<Mpwk8^huiNGZeBgWLcv5=~Me#Pfyc$D<~|WS)m%8ie#pNQd|j=S4BqNNLMI#81=?ELnMLNl&FY^1~3W0lCh9OYDgqrVw=?hiQyE^T_qU}B=YjMy7<mgN)LnD6q1KXO>$ggUhEJ36`^5@nom-Nk2z4>|Dj^I3HlzGP|eR{@*a|8d=kBp_-9gFSrhimx_#iVRp+=IP?fEj_ECvFs6_24Ju9A-4z)Hpslr{X+A>%=^h0;clR>VdA^s^UEd~ah41T2mMPznto&>AYMInBwGN$s`qUJ7|X&o<V*4Gsn;Ibfl%)Xg+Mh`6D)zmCYt69YW1`8Y62JtF!X&a@b-%JJo)?)!=h<X5-A~G<0D3o+#>;W^^!^4Ey%Q`7DkwFm%Xtr!xVP;_hX>$lVIq|*3R+A(RNkemu<}+HEK_o-WTMC&kjnte7m(jHPz~!GTa#m(_kOEERI+k)91QxrW)+4PB>d$i{cJHH(QedKZkBCtyB2Fylf`Uu+TosD_8}uz8gGC3ORQB@H^jM=nReC<fY6U$nHI7!aS_=pbn@4iU<!Y|k)28A%#)uI;Wj=w}B^`cCae+)#11))txRaW{$D~)aBZg>S@mw}ZBep)SwqHrn9I;jnt-U57Dj7IlId7vCDAeJw$CH5cd_N0&rJji*N*U7~Rjo8_3;#4jj7juhRm&**9w33x)47=n5eWQH!)LI3FCoy6Zy3`F_FQ!t>@brl5=EuTDHAj2I%-4s^ViSQZ}XDH5t1Y8H;px9!l?P-4_(wchv(lilRPE5aV-kn<)q2}d8$yIG-F@^4T^~jr^qQtD7l=fx@LNRYyKNiUeUNIhD%1rox43LUy!CiXp*LhlN@y<a&;*UIJ4kt7pscgHB!noF$HcJrxBm@N{|Q&-aufL5;sa0znax+@S@seHTYn+MuH!nM#D4Lo{ub<y0Swg@e7?d)7jBkmdY8+KZNhInw%P~#z)byBuk2?Q`#l-Og1DWy%U!ct;Ro<&w&bFDj=O;M(Y|sBCA;Au``Do;~VpBf=G=j<M|Q;VISm02D@fSTLNRUM9BvyXlU27>zV6SMd!3mK4m*zUv@02oi4_~=-Sl1Vd89IEQm!SQXyBu^)3nhl9#T@V=4ffxZfEWuT1m+3tW$fOHOJ+&X8rH5*{qq*bwt8W%)m))nMg|#UZr1;QnCWY;ztey1u2Ncf@zIge0#86G@l*;6?>Sxd@XbUNo0h%uS_FNb0C+K@L%dPl+B1us(0H1`nV<S_-9|G8mYWrGY8%IBr%Y5%{$n<ibr?KQ$atc(t8Rl8`{y+)A>)DsI>^lR~9o%1#%LFfn3YeO1qCT_l+a8W&3}w?yk|wf1msZBJI3g{ljb5)4ILzGZ2)D>&BQ@a#0rhA381%>Fw>x2SgG3;XYCa#$;bT1fIgq&80wI_{qOpCP~5<NzbYYn4RL5(4_B@xKb^Lv}+AErB#$U<3v+IpUXo^&`>8Fb=r(Xtmi!zPwtERt(o?Xcpj*CcJ@K+(;l%vY`r++JX89@}0TQpXJ36J#JJcF;+vVSV`ex2Qvi_0g8n<c_PL7&W8F}`frImu~cj)Ko5f`c(rgIgrY6$GcZRZ5s)`xZo%nxWDtxO(q>e83ZR@e$gRz=5PegyC*o17G>tnfDy}nEtY9Y%KF~3u`2v3YM}#!2;Z*7L$bVU=z8;}Hw0r_Xoj@d|9FPdt3Nk+okJ>>30&sAxD{l6|-1@o0L*I8<xXFO)as8chJ-T$m@`_<CG8U)&7F3Un^qERuitQB7L*A@yyaA=J!N5#LKaRP~9pDJqv-X7fZtiwry@qb%DNSNm@(CsA5-5Z)BC4g3uAKx}>JprW?S*K)4xOZmHL|J&glM4iwWwBln%tpOLu5x(1PO4_nwDCafF5%jiFN6uxu!~4G|U_1VZpOTt4G<$gVU>>ia~Uqev@Mnm&c$~UAKbvaZa)xXkKg$R<=r2P>G&ieKAT<W7v+K5eAofjo{7QU405xj+7RtIKY4fn=~Sh|BW;Lt#Zgw(4PVjwtH7}4J#~51gP?FHPly%dB^VAQiOe&@e;3hz6YcM$wL8U(;b}$S}2P0+Y-bHfkxXZ_K14Me#ItdF!5&X*XJNH2!GEkbS&N*C0qLC)eD6$R28^TXqD%Z<|Ioi1F8jDm6f9-N~%A+d=>YkUCLmH`pjy2h$q!At9a1tyJbhbGWx?bP2L9dQv)3$BY-OgB%@Sns!twQiMqw$x<+S5B*!$D3he_2DTzJ+@d~5PYUxV<*1mQ!dO$%1pnHxsMuz3mS$5&7tri1L&BswxNmXAhJ3lYQc1hLwi&PN@Bh$=Dos;*eI2PC2L|q7Ut<Y#aPk+z&yk3S-*@aO1;v^ljWKBdsfFHJ=&9@m`Rg|El5Gs#p^m7{7#$mOlrN7yR0Pc$tkEh;{CM_sdZcR?el~E=MA@q<=pchpk36Oml*#sN~VK7lYPs}&4xd<z9cR)*2L->il?7(-@w116dc8Pto+BJIKO3b3^cP#CcK_WcQC(oO>c7So9O;q;h<EMXS5~(p{WA!E{t%-dcrkBeAEUFo6ck3#jZlxMEJgQ>Jw)#oa*ZKZwt+lc9hKeDb*X2gbSM<T^3V!->rw@>>>usJJrS>~1wo`rYY*s<%Zr*XX&Lv1F9`p(tI}vOU@wQRI9;!%kIZe$GOc!-J<-sX>x`(31kr3t)njX~7$dq_!tUW+VUAC$0<o~YTJ(t^F*;kqs4v7S|?{S@ybk!X}BL1Q#0Ew})kyu(wWfVhzhi@UmSL!O@$RvjtPfGhLH7N^YVNs|`%N%3>sUkbeOf|{~g#}(VHxg_3wx+sL20~BQ)OkY{9>V>>!6-!*L)YW-&oZS8l&ztt$nZ`j!sSevC{ZvyE-VWs!p~R{O1XM^ZfXe*Y~;zqzi{NNj#==OK^CV1Dl{?=X<IlhE;A=Y@g;0e_khLJSON0|<Q+p~_&N$L#o{(pE!r!>O84`0({>r3WBg#<u*5m9{#cv1%XzX?wR+5w=tvs4Z3+D_rY4E)(5^XE1Sb_3nuvUMRRlIx-)rXK>i}MIrPsQqz;qNdFU!r!H`FpdnqaP22c=cYI91T*6`-$RQR$Mv)pgj_X_Qx<nX^}cV>-;5f=jridl^V9gUIdaQy$P2x%m|!O!5)+PFyy@S{yN~QsW7J)CU=|fGvQhR5RH`ePLBUf@3R-#`?fw5db<duC&O^SWn?;!X<iQSgW3L>i2^nKUmjEUpF*B5}65IdqQ(`H7Hehgnt7v083Ful96-;Pw8=i1ICRv;Fi9*pUqjW5w2Sv54V<dCzY|)E;6M60nM0`70p|8%&*ee04ECvzzBf?SQRU`U<XZ6HP0bh;z27i=a(i}5Y~R6v>Zy&@v5+g)|U)C9t25HcP;yyAco3*rlhlS%HSp6Mc_=<&bLZi>&k_dgIHT!Z5Y%|0X5Z7sZ|9gOCvmBX?@SA)CO4|RQ-yU7!#VEaMuQCP7!Rzjl|!I;777lDr1U4TTReI6qpiC*<=ph*DhdXf`+*SM=g0y7v*kaTlKKD*}R3cBX;7m_1T5o1rqy+HgiHK-Qg<=JvwgyWpzD!BEJ^OQ?qj~6*N+Uc!Zc&A+QhdsB6hE>Nks^cEq)lBxs1|>#}Ba8>Xgm?&JwV|BrL7<jddpNm8C}MnaBa&h^aIHXehpp(=t`#RAmB0W4_PkVCkLS)`@C0YVcXE*DBy;LE8#Yra}jdCbqVxLss<;{S?4Cp$|YOyZ4`^sU11GNQzeo>X^MDPTJy$H6TLR&toVn<NV9mIoDcBCBg1A(Ua+^1fy1ZhF*lizZl$*+t2(MPAx!8)6i-BELB6^*mdpf=8G%opVjY42{E0;4c)v?O0JV6kNuRv{o0hK3)p#H=j|eL94WfCEQw*7Bt+3uIOD!6szh#&#vE-COZ{ytDD8IO(M%fLu&Dx&AKa_%uv|R=}&R^Wr|eDY-S*x038bJlKx2f`D<F_=X97l2bh@K6OnE3w*Uc*CD8j+nTtFdmkTuwjgZoag8a~OS=31iJ=EO@7OD#ZQaWLknBzBJT13aqs-GHo3Zi2Pp@)8_$U>q}7rRR}he=&VRv~h*g6n>uk@sH5TE=T2A@3N}PJ|~E(?G0Y=!{{&4%1p%by61_9z9}WALSPqVC99A<pLr8fHoy1EhqesD|hv9Y7yvA4T+-(J$p|ph2O7)9ln1-#Z(|ZZO!E?c~f8az{CAhx+WWjjC0;E!PdC2P%L3Cn^<k|-LB8OF*QY`SSW2MxK22q0k&D?Kqa->8YKEO%qgC1dhNCqnQQ?qOaHRm%WMc@K&yTnWi`S8RI|vB02gKq<5_f9L@6<>hO;*SrRec8KO{+|v1<c3Jz>V3fNtM;5y%oioD^S+NYoo4S^hOd$-??i2JjS?xnl$&F`la7ZeH#wC(K!*U{pq?5}Ht!(`GUboiKsn2v}5qJGRUhE%U8qFn~~<HPTn*JTDjYG6w`I$_#PBMw6<UcgJT?pxUy;ztD?ns)Q^r86uwRi#GgG%TH7ewKf>FYQ~D@P6E%2FymCoh86%})wanMowifb<0@PMMD(3jRL}4qS9}HR@`5fhT5kc-Cn%uJ(h$_iy|LzY-X@5~xdPEz4(m3xxu%_Z7NjE3Z1dW-HP7Z+=`u30mnqFSy4B9Z6lp=xWh%-Z^I6y-Q+ACCWU32t(cO$H?M4w1XRDFG@O>tV#cANnF}*BgIM8%oo02%QW=|-TVrGT&Av&yZED_&+OV)UM`fhZYuj(@0tQ*$rtnFr`t_?|$l>~gM%bu!UrloG`YW*qihv&c0pb-yEF><a?Mz}F=&}=5ZUXMJe!YZkEv`)E>;LfK;aBDkM$2P08IeSZ7NDv}?6B2to0KYcS=xPre@i7wu6yYVK62p9(t^FD~Lut%22z&}Fu*yZ>CQ8E^S%5OTQm%Td2{PChT)8NtzD4t@uaGvls}av-mmB4V3Ca9%;fKzbnl23xQMQ3|rPsOOt9^3VmL<k=J&Bk#LN(B=5^MGrC0+x;7Gs4>8^7>ynw67{$S-uE2rZhQ*5!CzK&Ll3V1U<iF9{<w-B&zzy*kf*yhc{8mQ!CSyQliqYkPxYO%Ixkr$;@F@)I7YllA@>dtIAA319C6>Fi=1soT<9$6~d|@az(})xDutk(Z3CG1Lp_R@36mD>94@_|((V5DFX)LpwDmE#cj&i;PLnQo^+oQ*nZPGB#`22!5qetqi)l{Z{6EnXblK&Jqisf}N<&Wd>_rY8y0Mr30^_OrG<~en_DPL7Ll{-dP6>HN%YI1*_6EZ5FRic0D=i6~Dd%RC@6Sq0(38yEYWzb)@gpkf<TbWysYW?krSv_i93hYg`S@p~^K^;Y}=<?y3EeShX!7-@;JsB>s3VDSEKhuX@F6sX-Soek_i$US+ov?R29R*(CKNnky76+aME90ykB{rZ%1Cs*G{J1Y5L4QzGQ{sJ*%BO*R7s5+ox%VuRCNo2bzur9LIj(qx>^sC!gpeoc*-D8x!y8_*$tr{tPqbh8BqV-y|vQt?@F^T|RfffuP13}CKln*`#j4G1!sqgtVssIFfMvVB>SSI%*i#!?u$bR)`=h9N_F6dFS=Y(WRf!(2&`^R`|~l9XC=1@c?~|FZ6RALLReWJ$`o`zRo-i!0i;KDOzxWHn%jun-v1#{&=q@bkH%^B#z<BHqcw@a6ND(b{^*JgD;uKr>e9Qfz*)r)L*f<Z>c5$zm$F^Q3<0r5xVceYIB@`F~K$rV{{iOd=RHaZ&veEKT^kYGf(J9bHzZi=3RL+XVxeFslL^b7}ODSzGY^5ydbnqak39GKJD${<E}KCCjrZEx%S)%d_^K>Obg;5>p_p9!PAxa_z)|r$CF-A`lI?XF3SKVTztI#u``mwOM?<k`Tdx8%_<geiCH8UGuM+OmPwuU@jvo=`Bk31r+s#tA&>+rDEmhv&gezf5y`#RLAzN4zM(VlPOhOmCeqs9utJe%LA?in)gLq<Z8%0!#ThM;zX42<e==HI?-_IfKC|$c1tpBTuHQ^T=S^bXz?MG<n!W?T^5^2pu!EWN=q`Q{P<6M@AT!I>7uJNZaZ9CY6<yT;d!O}?aMLKc?VtOF_euYNlx}vC-U(kk57{zDNi&$oyXd&Dry$F`83w3R)1w3repfO2vh!=N-ssBS#7e2hG$CgoO1tdYa@`&-w2|JXf<}3zFsNXvz1;z`j4cgSq9*$49xkPVIc-80_z&w7@BDWY2d81sJvW9E?z(<jP>|Jju($ni)STF6K7x)Y?i}i0%xWP_ep``Bn1+t81Up$;aFT|@<d5ACeC=x@OrEs5R^#^d74%q@tmq8naeh?qF73LF>Jlkoi7`ZA<U4hB3g?n%O2^H(qCRU*#gh;$s5k7)LR22mx`7u1SUm2t+my*#G+r-X`?}{f}h6rfiId!hBvswMV4oqdRZA^2a`Y+4Y8N3MMJdZ;42hOErBbfUD8$+ST})6L43s2a|Q{Q<O(ZaOzo|laXuHkZk2jB?88A%fhXG#L4vUV4LQjYWGv-BDjPs3R+cz0y()>B!XIV%qDVI)-I+ydKkgS9xYv|RxO~v*1A9ygjlu*4psSp`61UnoB8RGEmvUOL{JU!W4xjkrNk9uRJu%B($W;~DoY(RI0O&LC4Re+A@CGX{*Y2j}BsG~?YBuvn6=X$AnbqrywrjJ&fn7A~C1+bsC|S*<Fo0%B*1gvQmgdxyQAlg%vgn{LBWWGeh+q@QRDf1eP3YJ&4G>B_odS$0v?z4}7q-e+mDZq=q_v*-1xiHbH4?iwS5eYn5Vzv>h*GO(BUZu)REE&tO4y6yU&O4Yf84sakB?e?h;dV`=E;*fFh}<-qfI>g(Nu_)s0UODt7zWQLo5Dt_#p5aOQ7K8pV8E^(Mo)1Y!)nV`UcvC4~n<YvMz0SZ`FIN-e9Z7aWVGKm!-#A<7<@FudbW?*g)<)P%Cv(pA6RTMvmpzYmj*cO(>=m41{HF!ZCM67^@*-@?yZmW5y8`LZreCu@@Z6LPcyWW{?SW$%%Owfw<OW6U#J{fN#t;c{As$fwnMtUI!JAhL$dQ#jj+6hK~>TpX*jLg-U4@yC{NUC^d*CnWz?H4a@)*7p}mhpU(g=;2#u2QHf#|_i@-;gk6BQglhBNy~3(IF_e^&M-8C`PCotyeMS4aS7#+rqRIM7&(B<EcL!D=p|L;J2IPkQSG5n~IaZpY`;+Y=Ho`%>4*RV<i8qx)keSV>soTGA{||QnE?W')).decode('utf-8'))
_LEGACY_ACTIONS_6C12S_4Q_SECOND_YARN = json.loads(zlib.decompress(base64.b85decode('c-rk<U2j`ia{MoP)`Lk=5|uZN&D}9pGcsg*h0Q=143G^11e=FR-h%x1IFd+S-cwy&)#p(5IC{ILDc<vax~r?JfBEl|fBo(EfBgOTlYjc<<cH7iZ{Gd-;ripJ&v%=XhtrdP`|Use<v+jt&zHx4{Pz35|NXzdJpXd?<NL?|)gFHM{I_4Pe}4bd_07rY$=loelhbAM@y8!Gn-7!!__*1;`||PqkDKdHC#RRQkAK?S-2QxWy4ZdF!`<z>&u>5N|Kj4|;eSr29sBV9?O#5B*uQBp>Dw<S_nVKO9^3l!?cJvzAD?y~%^nU1;^XG#X8+c+`CGR?H+dCk$n>@Qr}<Q%2FzX;&K~UHt|gCivN-7L^S8*mKHOZt-9+Pw`m_B5@U~gI$y=ZQWICQrJ03s#dA}GA`uaRm!Pn9e-dxY$zh55LpEh^%MKu5HaP`2YyPPkgkGG%ai>O_kfBL_jaq!8kcWf%#!8sh@*(mM%_xAdEX>Pytv@<7Nx8`y`T<uG@qcHteI$dD@p~(R|p;^J?Eze^Q#%wYi&5X6*(P!*=-09FA{O)|`?T4_PreIwzgu@MNhVW?RXUjnsw2?)JPCj|tmg-|Ef0EB57{cch2Fy`5Z~7qa-m!c5a`t{i58lA-$Gzu=pT9{beeCbk2_Mpd?cYw`H1v1Vhp+Invs>jXuqKnk)VM&#{ObH{b++$|w_t9Mkgqmo#F!Smy}h~Fy#4g+pEh@&-rv0c=fg8$(BPF{Vl0vJJB~C5+gp3mo^TKC9Ff_VgRA`f!LR_o>Gf~S@4Szzx_6t}f1Nf7Fz*`kabkpng<J76fH4C11n$-I(zeWG-iK*#vp%K+2poIEAZ4x!e9C^1jRks2e~@_uqW#$6kH$?dI#BVTO17`Efv9hu&p+{W`dnWHcuIc`ddr6M0F3+nPqxNjzWH0=gxHpO`>dZ!O;v)My|7{Z`fKBVO}_Vm4Yg81?z&+R+Y0UBd<dg2X0Z5|Q}6B;AvMx*$gW!HkgV7bySGjbEdTBl+uqYTYX}jt-gPI?`?bs1pcidrShyV%LXnQsl(pY5o2cbMOooCzMi>1a^-Hl)f?g$qkwb>g!8?btz8~P~^=Dsy_7C{8I)F98)QKbSFod5%PUkj&5`^U2cQ+m^bLTXCrRX&pcuHRYGP8&(Ac%)bIqfG=^<GDoUGTx!{CIu$*QjITZhQkR5Tn>=sCIoR4$*WhdMF0%;IuKw9hsmDNa2IN>)6v<y+KDt)oxIxBbCD;0AD#+cKbED9h7~<Qy%pFFQThv`o4*Qu46EBjt0HYz#A$N=JtnMn$)Y=@cOg8AkledIX!=G{kYv*W9l3e9~X{k^=!m^{B(DH|HJ0)?r*@7DIrX0hr+i*8s>61+{79fG-B~^1T^XeK`86)G|b336ji-TV`QNUJRQr#np!7QtjR-|IH=OKK6VegD?R@?4QJcl$dgTr$*%)Lon79^d<2TCAnLdE@l!La79o0iYG@_I+TL=6@FsAzou8|~gh!*}y#}_vR+#L<!H&*5?V7WP!tp7@%Mu%b7%J#^Rh((+t7l+L<yv8A#pDvazrDSAOpAf0)$@NoPtce1@!d(;*4z8@xVOf~($T4zgN!0Eh_f;u>gZOG4c^09vDfl$B1BLQ#*!}u_7BKZ8f_?)ril3=T6|2suO%3&iyo%?E`98%Hu{+&WfFSZJeBdzO_U$uH4(t3^Kh)!M1(R<I8%p-1>L*A=zKff=;)iD7L{$lh8aESg91*S7C@e-&eRxRz$clRx5u)gF3fD%MJZwj%&!HG*{yP`3}&<1rkY5n(wty9^flvLcwh<2>;;*qK^wr;%&s~e$xwXlY=DA~+gm>PB8MB)vxH4H%;wEgw>|4Yy3=X4UQ7d1;06Ys?V?Tw5(wu1I?1+Xr1D*0iex({xu-pr1=~S1`Bcv}XZfx63WR^qX&lc6G9`e4G1G5g;$O6ZVK5bO8;f9M7%s*=W%f<Q*Z|f+6QXUh;jIyGJ0Ds)r{e#W$d>FK?L%qA+r>@BlDwQH3Io))JD(=ELEHbc&O0`&v}^27ijGnX@&ed5fZW;t*U~&dItsvz+ZOiamdNShUmo7Q|Fem~0{dD|9IwH?fFq*$Dg8{r#T)E)G+^-7hWP3I&0h{(D(G+_D*=9ug~NC6I<o6^mP1Lq>lImbx+Umx^a3Xgl8S&a7&#Zst*tOTIUZGqCv<4O;%Vx?o}SBW3xM?zyf0fUt+jgK)|1Q#Wh)5;jq3(K23kYN;gFqMMRQ7aQ7PeCT|AmsSpvW{Mn~(|?+top^I!sDA2Qjx62RalVJ@x$7CS{|TH7^=a{zCMIYF5AY1&D|RcHqiwVt@nFizG(4pq?S(<D~Y=ni;u3Yeu}Y64@H3@zFQU^8`F!6;_I_2)nnrz@|e8CUFN&{auls+H?&K575Fl2)+StBc<qGL^IQkhKiixr++xhrKr^TOa(Alsh9dayRQ<LnM!;kPZpvc!(1<K7(?|!)s&y@Qvw?86GmE&#Ag@S<HZG+3uwkum~Xeo+e{8P=cYg&vb8^aCp5CXHM}R;!%^}pG694e4#T57QnWKnps(tg&Lg12>|ikDnS7i<T+Qcne=buHwm!96V+^mS0J*mD<BAe;{1~XO5DoG=WTkJxMPVj292*3T5hEYIQSUNh-k?golZJ;#QEqF|B?%1s7KR%3|E2$5NGHH(e#Kh*GHuy5iu5vWD!`}vp(1+p=L~CHDcUc^SE?g7V%a2szbDn*w#a9xzIL{^t*inakeC_7N*)#$I+R?NBps|b~I^LZPbRe3YpEv85-WK`oVK-*GSSfxweJ*zV3?qimd_`>$lLX*M&7;cj{QdG5!4RAp0@g`AT$L3_|PCapK4y!0o;(uX*sYmAo#cumr&ai$`4Xbt?=qJuI%f=TVmgp++ykp9R7t=MRS!%Nhh!gE4!Tgh$1`cx~1}+%ucJR2hmUg?7#~9j@t1#ISp(fHZdzNrDHCYy(gNW!nMxcd;<;0Q}0f-SB%zjg!~!0?oZNZUl|mAaP8=7R*QX@<HdHveBe|Gf4rAQmS!l*C^7VcL2Q|?_$QeC9qYg2Ny9ra`9l$0MJqbP1W!&35v~orb94Pxh~l#J7(OpCS)xA=Jn;^N0a`6oZU%vJkmzcJN^gc+DGKb!@ew6fn?Ppt{1a7H?qys)~UH#G-Vif;mw1dLBz>K6jFtJEihqZ$)T0SyV4=@$tIvmy6w^YLc<&t<4kQzT@R6mi9#CCG|wWO<Y**SodL8g1SMOq)Eq0}(SuuLlp&YkfHp@$$X_<%B_r7R%d;vv!XcAy1u2PY3|mg|(J2sn^qyv$gB|NR{Q31%<XQtKPry$kKR?UCo{m&SUa0k_s~}2|%MpCs+Dg-&>Pf!JvQtx47}+-MhK$@@S!pd4USQb;)KoB(^}NuSyxgeIN$A58)5BBdY1RvKsc1u}lU<|hGJUR=;KNcTD9IgYX$)B+0Cz;PER=L|<OL#417nd<r-LmVPUz@dIH1LBrN<n8O??v%_U1T?s@=o;el!*BXRnfr?SySdd#k#v2sbN93{jew)<23|^C04MD)mXB!{qCo6XALnRCNbos7FgO3X4T`EHdMiO7{b6G0!xMv0{773D=L~p@L#v0$dEg=<L<X@~7+$I_n0uOHNl)YQ6e1;}->D)h6)3OL)r5t|Jc+>U^-)hz$bm0Ut>#ZvYT966<kMq2t3o0RMsF!9*8tIs@x&HP+qHtS7tFX(tTf%^MHXTiOG<%?wFlktCR_$s*ggV!J5zlQ9&h2}2JxA3JNMKJE+z<(NZ-U8RIasDamt3MLq+8oNDzs8Ho=X-<gfkABDz8esAaEKy5U6)@xqNzMkqS~$=ki*UAa@|uVrPvKEAwg^{6Nb=N~93muzM!Fq7N)V`oRHVe9JeOx2T@Vb?1ny8TOJ|&z<BPe1_*1?^=19j-OpXaqBo2F2s1J)7iTmT+KuJU)5H36eI<DR*YV4XV6TYE!<wH#alb3KymJve}x%Zl_`k1@sua~%J5?Y=bK?C5=vO}&=C&jW{!eJ$Mi5<g1qsxn(!uKXPQylQq-g3zrO-lc^zeon-3X8>*>i3wm4akj`Url0T1mN&uL=$FzbLatN3_CIhKF6f>8TwTyTA!`KPlz|yD@_j*k1|CLN_8-EMm#90zgm<s6BY2H6VqZe5=pX!?S*Q}PP<A}xeIEZ6bc*B{5=MV=p-A?z3C(er74E$xVD#Mu_T(sf^OF<8(?;U`}W7`VUUBZineT#jTzB8EA5I61><s2v^0}s&p5%u%1@_BvK+*qCZ|!`j5P{XcVK7_vW-*et_q|m0xRlfFAbdLFH887%AwkfsT(2Q9(2`>8Zt2uu6W^6)<UgVlh`EMHnDJZk!Zw<+#ydKwM&8&i~)6pa@EfQn5jjRDePV2Aozq75m(TQg1mrQrqP(1zkk)Z^$`NF`rbGL1-C3f=CkbmPEW}RD#G;8v89DuJKEI9=<ZJ;;wbwiy1IoP4KK`jF;ypyNA`QfZl&F<I0+1q^!deMeUImVW+@*-y&GLL*H+r7)dobBLaY>?+K!{<>*3VY>SHTX6529N<RZFcAymQF$`ph5?x|PV0QU*Yh;E7?jZ)KE1;l>QORsoIGT|zLkqv&JG~~eHk-YC>d29}LvP7I5T-2Nu7_MhNe&xk9X@sT};N7QUXo~-%RjW}#(F16cN=ixMif@kN-$dTbIre+Lb^iw0QQt&_5@C;G5U(Qxo<%3sq-3yvsKMs4q16zi@|*|`j*z}rIMF(&tc%PE40fIrUerjF)>7JG-Cq*NPA944iSSI998mAtxX|Q)q_&kLEvKjzwP&+_SJR_F=fSfObl=Qb6(6<gj&xSit>=q@T^O?bZ4!v3@xc@^Thl^cWr85#{%6~45bABH4v4LyxHxmjx^n7@T8M%cR!|X?rIS%lb7tL<XDLO6c9S)4<!gC$<cVHm-iLE%(NEim+)B~m=E)-^`N2Xym(E(_+1U~=xlg@w>D&*{{S64z8!xH5a6EzDtla9r34$S!XKB=O&@^2LbYLDX!W)cIpN44hqTZ~9MaM2)hK68u((vnyfZsP|YXMgg0Umz5X$z61&>4M#WOYTLPoq$Xn@IpANA^$2S{jzl-&9{hq@cb|jzPN)etRm|xQ=N3tH#$l{Ae->Wt<63%{N2eFDU&1zkt&LSTb)X%@!y$)g=PB&*(7&0kBXB-BMz{3&kO8P>t56Ev3+c1-OwsZ7qa5?`N@QT>(K2d8(a=WBZW6qKU~+QC4Uk_?;f*%+}J-(^biB`55N&DUY)Ko<g16>o|;^>Es_KpJ_C1rfPkW=36uKd-o#_^j8#q)gd)!leU&J0V#AL;loMJt}a`NMzdY{p%x{s3&LMp*8v`{JcLb)19b}|QpX1!8?;tI=;_(*VqT{_UP(@W8flIvPfN4jy*4<(`U<|R9x7Wi)09ZQNS1y~78Z!zQtcL&%|Q|kY-NL|z>^}}-@8JR6s>4_#`OU0e?`_5@>yi2kJaYosB#Gr1>`F5b7z&wUs)l96#~t17QFi!oDjwa6`W<5rG8eW^LaVy_c9{{+%D{%>?F^l(5PjuM*Ts^T7Rzls8n^}e#?HuGrG1Ek$rfgO@BiP%hM7FYOPhH{G*fW-IaIBY6@_OAahM<&B0DA0XU8AaLa;`eQ5)50j-TP3BZdK3bCbBgnEU#wPRc|?nqK`Akk{}2~SrJsiR%yVR^iZd9NgAyR2E2m|KgBl2#jI+^F(~sy=_Hdn8t()RgT@#geI?5ZmEnLiQ{Lfo>_`@;_XYne<zs_Mt)?A^9a-dW@aQjJv}sB-ZL8*6Kx2%!lGY3SnVLVaky#uM{13aSo6ROVAieVK1XyiDWAo^f(hCg9KoeN+j`m>{LR8<+B<azglZWCG7dRN`t$o@+D}<mcVNSaN8Yak6eT^S3s2sUzp{ITWN!U8V`0W4+dzOPiFDck!yK1tjeguC7FO^FkAXn^^u`xNs=h79CT#J%tLLeQgc8PqLG3uay=NTRuoqtS`>A(O<_OOilUB@h?G3|BV}Symrsu|ODFnRPf(-P7#fMbx+6=B9^)1&Nm%Wk3!<ae8rL$2>phb)vUHtWt<oT1ry0|wC#ST*n@s)Inwv1%B2JB{Ipn2l0n;`Tg^_{S(lpJD)RkbQl6?k_O4WX)>5-f;fjbvxq%N$KmUew&PhMp0jGeEgFkgfo0B$sx%0u(&1%C;w=sqVyxbuK7CD;98E|gpwal}<{)y()=xf%fB@k)g#+4T>}=(_{$_;VdgfHakYwHzO|ln;+;_RV+i|J-foT-7s9#FFLgxZ<Z;+jk=)>XJ>c2qwfwn=++}PRzcu+sR~Q-+ICTR~=+2jpilPAubn^c^sWmzt6;!v38U5l-~^xb13<Nxp;1u6}c39&a2W;FmTM%5za3q@7_%jrVCW~sJMK45zZ-^`4s;+fHRk2knOCo4ouakE(`fPLqBcq;Ovv*^98M(G*Pt3Q#~uP1u>~5v%N)*;dttxhi`)8_Zwsg77Ov-Z(K*Iq8CbYfnxF&3&rEG^)QL)DhH5tVk4R8v2hJtMpMj~U>@>9QtN{DPs+kz2cjg&jj)W2oIwr3*Fm-P)u`U{0l9a@a@(|eHnb~Eu(ja1XX%?PM>-WO(Q49Y>OYapio$C$0tixozM#8gi21N;UZ?>{*f`_R34`t|D0VWVA;pV`0b5-DMit%5aa$NiCLkUKbi~``ShGTpHx@Jyojd7CC~sbz6$(V)4Imie^{H9*;pFLEM{=#z!>TUJqwF1~37yrRODaVL8A4OlW;fK`R=SxNu<c}fo7QFw50Z$1G>`pIr^N-N`_e5d6~0I@R+r_lO-IE~z85EE<4#&i#QDx-m~S-}9lHqN<3C~{s6(l&<*|^v*3AP(+Pu+c;fIAk2}RlZATt-sW0K62+LhO8Ai-82laY&hSZd{!nmGZoNEreun^dhBRV%~7&%4Z_)@9<wj&<s7aS3?Jx!L5fObGx(D#_|{p7aZIo&eLv#%7rLap|0qt;r-M&?cU+UhKL9{z?;Z^~wo)5Q4in<KRnT;xkHNBpDJ_sM=UHe8Sjv%@HHn&3wS~GF!k%bhpu35xj<yLps4&cSU(bJQ9Es3@FOCkYJcxPL5;cC62p7<z)I+sG^gI-ZRml&@wQNzBg9(Sd#hX0x?*0Pv^Rw)$+hHJc$aB6COo)(eY!gB9xXu!f2P^Z^l&pY$A;I>-o9?huaJ_<)eOS)uGlXPTH}>r%)X93ri4#SHsLS?Xndnv60ObR=R0{^pr%FRsc7mjHl1eRaj>`N2n<IS0==Z*tiNWMhTz--_FLUfDqDlmHc7r_yPE>uP?{0@AqQasH!A&Vr7(pnwP8V7kweCSyC?4UP?ACa+m7A3?^2PgUZc5FA0^)Rp;5-84xz)wiuInn$2)<S|oW|Pi?D#5qf5zm6m!h7Ej6!83$epb2EJBk<|&Trs5|zL0}bz3wTyUt+FXKT1>JyS;oT52haO)fr;Qjgsy}o0-kJ|FHYyk)O2teHVfsz)yfsNYZ>tDcxTaB9eA<D%?U+NM1+f#Jfx-E>BVD3tp;UF7WS@Dk2ENAyw!-Ei_#?eO2y(~2k~%gmoJ@NUN9RHDqSYIaNju?mwmbGGS!{IBAV=2rQYYd)H|6gOHV2>jlJ{u!i#n_LO@K)Wt0TskO@{k$k#P1E2FWBvkF&lbCO*SkO)h6>FLNSAv7Z7)o-qp3$M`=`L@g1b~1ap;=5KCs)*>dhYGT<c=6KC;*L=&!w8}xZ?DKfVz<xW6URnB4?N>VTfy-%vR<OI;5*moBX?cFex0}QJp2$YME%eC7VBs4wb^o#UNb4B78l|h&gC--8SsOQ=GPV3ds&m5tX!SSN7%<TQXgZnONeGdcOR3VON{U=cam2m^@ZM;ZEghc5l-8#W}dl5H>FEFS{U#v6N}etjvx+#>Nk1?wp|(=(~2h*f@$79Bb6j|Z;ebON3u#=f?OP*Mxjd-tMeL_$CL*U8KsH>zAA$_z4pa@27{8!kYy(^7xpo8;*uh3SEX`#h^(wtiV!O}aHHNPZM3V9vqpBSZmW6~;qp1MZ61JWs$W$zpOLS&1MsEWcel})hy-;fP>y`G^O2oSq7hm9QwjmsFHIRGml0FvO+E<Eb0t?%&`Jy>>OTaD5fiahDXsHrGBT@@D-}@iYsrRa|51hT$U!Z!T*PA<9s1i-aR~HFq+CU1JXSVB;~8qYf5G$%R8x}rxQ@<%j-^4tNc6OwP>nUq%}co8`r;zV0^sMqm>9;_z9PRcNaA*>?th!6`8<5*4Mw?UM4#`3z4>&f9+keCh000LrI76sOCA>3ovP}eRo;Ns>|L8Vn1a8ogL4)Qv{<0TL0y&AHMW#U?$GoCgRMI~eqH8ZR%d+`1*&qCsI}@;a2b+^0^c#0SUES2z^v&$?935^TG<;VHcb|#O4Co)Y+iloIsDstgrZk**~flt&8Ze7r4>-Kxjl?;uY;u$S9%wR;YuM2u8mb51n4h>G(Sq*b>YcKFNvVd7sfgd1xw_@2m+Eh1pOx{7wZIu^YM)EGGl!qfU~~-#5!CnCz;5cqnJS+vq+4flv<vX5y<mdl{89Eg9PTD;67hYqN=;@Vb60o+ckLUu%Qz^r34rnhG!RuW~@1$cEXTwohChuOQ(SYjN2@71!Fp-$o2Q@3oI?eWnB&nXgqK2(#vv@_<T#zfOTKk4&W2GsYLy;ss1u#zSqSFa!~~qgt^eR-S)Mk>WP<`a8a)j0)(&%*yjZP#km}D?5|FRUXPD)b#XVtNjoM0x<as~`Qfqnn3hi17)~jpW?uN3(`|&Mo!5djmlVxN@*EjFVChY9h3tazVg^h&$%`d~=CYJLpyvL|3?iayP&`F?v>?P7L06`Kr`VR1Fl~uUw8Makb?P-?%W>&dX-THod0W3YkasF%Yz~~l(;h`k8W&>%IDxtntJs_x*UPIu1&YIVD5PxUCC(~hj;zT8wpY4Pnr#=4JkFkAuJOjc&?F6aR@<7()S1O;6%EUBZBioEN%Mon9KMktE0tLT{)b%)sofMtSoo}>szWMMQCvGOk1#}+lDWJf52EGkYn3#(Vw9Y7u&nQb!kr$~URA5Ej}fShFMOeBY7p)s)T_3KD(R)|%&PBF=g^r}ewLuaQV-42s=Q+GMwe2wD6XzN*jarWQC(B7P)J<mxsHbQW*x}>k6qhFF&7FeP+_Q2@g!j8GWz~@(h@b{-V=Q}NGNg%?!b%x2Z_%|6ohoWWYmQ*nvm79wy{LQu!6jKfZu6+2DX%(IFDso$9wQ{iuD#WZB$i;wr8d3G^*a3zPSwKsz&_-o{}gRkE#}=Dvm3I`Xy2Pcxg^>Wbt^p8ZEpW#NLD1<sMBm!SY*$hEhcng@*@H@ieqet9U}9Dyu%6o0Phy@;B+6=_h+u*0(Gp$Cz|{o>B>Y$7jFlqnG6;sg@@ru$9UiqFC3RWcG0`P)wa6J7pO%$@)r^WbB>L+i{dvY_6HASVzQts9Xsx>L@XiiaGyhW$G1!tf@AiR5+lPZ&!<JStU_L`4a|J7?rawZGO$dsXl6hT$HL>%#nQ4kt9o8r3UmN>7ZhggS6+;^?2HW`WqE&_0>zskx$bjGG6=0kp4o}5KFfjiLE<$K7tsxl9(_pD#l<nO1KV#)nk|ueYCYVl!L#e&@H>kZTe>X`c;Ley!V#@=)^o<w9JHLg!P!xEl&1s4%9`)HOkw%tmY>y`uJZKSXPmiQ>9*(%R#)bq5Y8AMQ*b1g%M>`mT-R|4MV7IqcTgeK~@6=b>T_P?dAhgWW2080YjZ)s+4Z7a^0>5A&DthL5mJE>V_136hJC3r<WD@NaB@R4cqj%)CGb<%ZJ7s6l7A>h5@RWG3#MaZR-|D$lwrKJe^ki?5t|Zn4U@UCMyoqz{NA8*eWHev^v?ZXnciA?mTt4k~n)W5I8||fXUR_g;q37Nu|i@dud>&lSbi`4cR@2KuDzg$lZmy9Ihk#m{;M}Fc7Ab+eLz`CUeruhld`y(`IV@9;r}nlny{t7jg;1A=<vwuqBRemWiqA7(W6$BW8q<&(dNu>`0aM0wWOJ>D)e7vTwc7Vt&*-A3#1oT8ZLBRu_~*Ily0It+hXr99|QA{AP<8b2kV!$tqltu%3$bMP)3=NvusGE|!v=7QR-%bw*38jG2-`YgVF=giMO1cWF$)s5VumWdW_N){UnQ7q+|y!GBk^LdarqjQk`CKN>}pQ_30x?~jyjiI)GdW<;sBP0MaGci}z{4R<adozW)b|49OVeQJuUvVTLUN#ws2ij!K2Q}>*cZAn>XQk6n!@l**&#LL$}tVG7Fcf8wOz}bW}J^(oo0jGgz8j8Bh!>d@pDrjJGq>{9Snibu8TyDu(tqQH8iZW76-q1ZED=!yhtZf)Eb(vFkC_gPToIEXx?(Lyf7dNU#*@e5!+)S1l-x7EjZAamBmEl)sT`^IFHqo+8evb>-4b6%bYeX>VYD5hiR7`nL&14%VdXDQylx-k*x;%^Oy8@K^vI}(qeBAFV%h6ELmLldg!7O_ERf#B><pjNkm2+H<$Og=AF(~}u!7>tKgqhn$2GIPMt-`BR>pZdVX|TOggf{_GGptps_$cMvf_sCBm(eRFtE2}_jRKC5+p87@+FC~H%!$s3e1n$iz)?_M3jTnGTFqlbev!nUMCv>TZ(-JwOaUcZ|GGHO5>J-qin3{L&cuOhFbmX+vQj46;!bpS@N?l9?l)WQ{BZs8W9W-K{ohQ`eET6Y<ab}1xWA7db+!#Qq<!ErTt{jfY0rMl_Q+Dy3h=ELXc(U1wzac|7id}G;k8vqYSkPvqS1^sW<bD*wpvoKk3*4Ed@a<I5MNAbXEf%Xad2~QfQMO9bCHTkm0V_q1ZmlJd8d)gnU)RCm~n#Z#c%zLTb&Gi?p`G;{GJ@~l37k(-JV!T{0-eVaIw$K4C}>cHi2!>S~BEp%zF#{2J(*ZcC}s$H!$lgF`=v%@BX%ZYi>6Tf2C*`_Pi7rYPD<Du)y1nwhz&Fim#Nq8Ll+zrTbXT-R+0$1<@x&hi8q7+*SM^RGNxxv3x8DMAx1eOrDPnM*2$HQC>{U=<4&_PE+ff+1`CZ9%f%w<1gEJf0Ng4Jd(|QI5rRe3!QmfDg')).decode('utf-8'))
_ACTIONS = _ACTIONS_8C6S_3Q
__version__ = 'Codex-Moon-V139-V138MiMiExclusion'

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
    0: {"last_step": -1, "debts": {}},
    1: {"last_step": -1, "debts": {}},
}
_PREEMPT_ENABLED = True
_PREEMPT_FRACTION = 1.0
_PREEMPT_MAX_BATCH = 12
_PREEMPT_MAX_CLONE_DISTANCE = 6
_PREEMPT_MIN_PRICE_RATIO = 0.0
_PREEMPT_MIN_FUTURE_QUANTITY = 4
_PREEMPT_START = 120
_PREEMPT_STOP = 680
_PREMIUM = ("STRAWBERRY", "MELON", "MILK", "WOOL")
_ADAPT_MAX_OPP_HORIZON = 6
_ADAPT_MIN_EVIDENCE = 2.0
_ADAPT_DECAY = 0.999
_RACE_STATE = {0: {}, 1: {}}


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)



_KAWA_MILK_SUPPORT = {"PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP"}


def _kawa_route_label(obs):
    shops = list(((_get(obs, "town", {}) or {}).get("unlocked_shops", []) or []))
    if shops[:1] == ["YARN_STORE"]:
        return "6c12s_4q_first_yarn"
    if "YARN_STORE" in shops[:2]:
        if shops[:1] in (["BRUNCH_SPOT"], ["SMOOTHIE_SHOP"]):
            return "6c8s_3q"
        return "6c12s_4q_second_yarn"
    if "YARN_STORE" in shops[:3]:
        return "6c8s_3q"
    if _KAWA_MILK_SUPPORT.intersection(shops[:3]):
        return "10c4s_3q"
    return "8c6s_3q"


_KAWA_LAYOUT_FALLBACK = {0: None, 1: None}


def _kawa_use_legacy_layout(obs):
    step = int(_get(obs, "step", 0) or 0)
    seat = _seat(obs)
    if step == 0:
        _KAWA_LAYOUT_FALLBACK[seat] = None
    decision = _KAWA_LAYOUT_FALLBACK.get(seat)
    if decision is None and 24 <= step < 72:
        farms = list(_get(obs, "farms", []) or [])
        opponent = farms[1 - seat] if len(farms) >= 2 else {}
        counts = {"WHEAT": 0, "MELON": 0, "COW": 0, "SHEEP": 0, "PASTURE": 0}
        for row in list(_get(opponent, "tiles", []) or []):
            for tile in list(row or []):
                if not isinstance(tile, dict):
                    continue
                key = tile.get("crop") or tile.get("animal")
                if key in counts:
                    counts[key] += 1
                if tile.get("kind") == "PASTURE" and not tile.get("animal"):
                    counts["PASTURE"] += 1
        decision = (
            counts == {"WHEAT": 5, "MELON": 5, "COW": 1, "SHEEP": 4, "PASTURE": 0}
            and float(_get(opponent, "money", 0) or 0) <= 12
        )
        _KAWA_LAYOUT_FALLBACK[seat] = decision
    return bool(decision)


def _kawa_actions(obs):
    current = {
        "10c4s_3q": _ACTIONS_10C4S_3Q,
        "8c6s_3q": _ACTIONS_8C6S_3Q,
        "6c8s_3q": _ACTIONS_6C8S_3Q,
        "6c12s_4q_first_yarn": _ACTIONS_6C12S_4Q_FIRST_YARN,
        "6c12s_4q_second_yarn": _ACTIONS_6C12S_4Q_SECOND_YARN,
    }
    label = _kawa_route_label(obs)
    if _kawa_use_legacy_layout(obs):
        return {
            "10c4s_3q": _LEGACY_ACTIONS_10C4S_3Q,
            "8c6s_3q": _LEGACY_ACTIONS_8C6S_3Q,
            "6c8s_3q": _LEGACY_ACTIONS_6C8S_3Q,
            "6c12s_4q_first_yarn": _LEGACY_ACTIONS_6C12S_4Q_FIRST_YARN,
            "6c12s_4q_second_yarn": _LEGACY_ACTIONS_6C12S_4Q_SECOND_YARN,
        }[label]
    return current[label]

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


def _planned_premium(obs, step, item):
    actions = _kawa_actions(obs)
    if not (0 <= step < len(actions)):
        return 0
    return sum(
        max(0, int(order[2]))
        for order in (actions[step].get("market") or [])
        if len(order) >= 3 and order[0] == "SELL" and order[1] == item
    )


def _town_drain(step, shops, item):
    drain = 0
    if step % 4 == 0:
        for shop in shops or ():
            products = _SHOP_PRODUCTS.get(shop, ())
            if item in products:
                drain += 2 if len(products) == 1 else 1
    if step % 24 == 0:
        drain += 1
    return drain


def _race_state(obs, step):
    seat = _seat(obs)
    state = _RACE_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {
            "last_step": -1,
            "inventory": {},
            "prices": {},
            "own_sells": {},
            "shops": (),
            "scores": {
                item: {h: 0.0 for h in range(1, _ADAPT_MAX_OPP_HORIZON + 1)}
                for item in _PREMIUM
            },
            "evidence": {item: 0.0 for item in _PREMIUM},
            "horizon": {item: 1 for item in _PREMIUM},
            "policy_scores": {
                h: 0.0 for h in range(1, _ADAPT_MAX_OPP_HORIZON + 1)
            },
            "policy_evidence": 0.0,
            "policy_horizon": 1,
            "policy_adapt_step": -1,
            "adapted_shifts": 0,
            "adapted_units": 0,
        }
        _RACE_STATE[seat] = state
    return state


def _observe_opponent_market(obs, step):
    state = _race_state(obs, step)
    current_market = _get(obs, "market", {}) or {}
    current = dict(_get(current_market, "inventory", {}) or {})
    current_prices = dict(_get(current_market, "prices", {}) or {})
    previous = dict(state.get("inventory", {}) or {})
    previous_prices = dict(state.get("prices", {}) or {})
    prev_step = int(state.get("last_step", -1))
    state["policy_evidence"] *= _ADAPT_DECAY
    for horizon in state["policy_scores"]:
        state["policy_scores"][horizon] *= _ADAPT_DECAY
    for item in _PREMIUM:
        state["evidence"][item] *= _ADAPT_DECAY
        for horizon in state["scores"][item]:
            state["scores"][item][horizon] *= _ADAPT_DECAY
    if previous and prev_step == step - 1:
        own = dict(state.get("own_sells", {}) or {})
        shops = tuple(state.get("shops", ()) or ())
        for item in _PREMIUM:
            if float(previous_prices.get(item, 2) or 0) <= 1 or float(current_prices.get(item, 2) or 0) <= 1:
                continue
            delta = int(current.get(item, 0) or 0) - int(previous.get(item, 0) or 0)
            opponent_supply = delta + _town_drain(prev_step, shops, item) - int(own.get(item, 0) or 0)
            extra_supply = opponent_supply - _planned_premium(obs, prev_step, item)
            if extra_supply < _PREEMPT_MIN_FUTURE_QUANTITY:
                continue
            state["evidence"][item] += 1.0
            state["policy_evidence"] += 1.0
            for horizon in range(1, _ADAPT_MAX_OPP_HORIZON + 1):
                expected = _planned_premium(obs, prev_step + horizon, item)
                if expected > 0:
                    similarity = min(extra_supply, expected) / float(max(extra_supply, expected))
                    state["scores"][item][horizon] += 1.0 + similarity
                    state["policy_scores"][horizon] += 1.0 + similarity
                else:
                    state["scores"][item][horizon] -= 0.15
                    state["policy_scores"][horizon] -= 0.15
            if state["evidence"][item] >= _ADAPT_MIN_EVIDENCE:
                ranked = sorted(
                    state["scores"][item],
                    key=lambda h: (state["scores"][item][h], -h),
                    reverse=True,
                )
                best = ranked[0]
                runner = state["scores"][item][ranked[1]] if len(ranked) > 1 else -1e9
                if state["scores"][item][best] >= runner + 0.25:
                    state["horizon"][item] = min(_ADAPT_MAX_OPP_HORIZON, best + 1)
    if state["policy_evidence"] >= _ADAPT_MIN_EVIDENCE:
        ranked = sorted(
            state["policy_scores"],
            key=lambda h: (state["policy_scores"][h], -h),
            reverse=True,
        )
        best = ranked[0]
        runner = state["policy_scores"][ranked[1]] if len(ranked) > 1 else -1e9
        if state["policy_scores"][best] >= runner + 0.25:
            if state["policy_horizon"] == 1:
                state["policy_adapt_step"] = step
            state["policy_horizon"] = min(_ADAPT_MAX_OPP_HORIZON, best + 1)
            for item in _PREMIUM:
                if state["horizon"][item] == 1:
                    state["horizon"][item] = state["policy_horizon"]
    state["last_step"] = step
    state["inventory"] = current
    state["prices"] = current_prices
    state["shops"] = tuple(_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or [])


def _record_own_sells(obs, action, step):
    state = _race_state(obs, step)
    remaining = _projected_shed(obs, action)
    sold = {}
    for order in action.get("market", []) or []:
        if len(order) < 3 or order[0] != "SELL" or order[1] not in _PREMIUM:
            continue
        item = order[1]
        quantity = min(max(0, int(order[2])), max(0, int(remaining.get(item, 0) or 0)))
        if quantity:
            sold[item] = sold.get(item, 0) + quantity
            remaining[item] = max(0, int(remaining.get(item, 0) or 0) - quantity)
    state["own_sells"] = sold


def _adaptive_horizon(obs, step, item):
    return int(_race_state(obs, step).get("horizon", {}).get(item, 1))


def _shift_state(obs, step):
    seat = _seat(obs)
    state = _SHIFT_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "debts": {}}
        _SHIFT_STATE[seat] = state
    state["last_step"] = step
    return state


def _repay_shift(obs, action, step):
    if not _PREEMPT_ENABLED:
        return action
    state = _shift_state(obs, step)
    debts = state.setdefault("debts", {})
    due = {
        item: max(0, int(quantity))
        for item, quantity in dict(debts.pop(step, {}) or {}).items()
    }
    if not due:
        return action
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
    return action


def _preempt_shift(obs, action, step):
    if not _PREEMPT_ENABLED or not (_PREEMPT_START <= step < _PREEMPT_STOP):
        return action
    state = _shift_state(obs, step)
    race = _race_state(obs, step)
    clone_like = _clone_distance(obs) <= _PREEMPT_MAX_CLONE_DISTANCE
    market = list(action.get("market") or [])
    if len(market) >= 10:
        return action
    remaining = _projected_shed(obs, action)
    for raw in market:
        if len(raw) >= 3 and raw[0] == "SELL":
            item = raw[1]
            remaining[item] = max(0, int(remaining.get(item, 0) or 0) - max(0, int(raw[2])))
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    choices = []
    for item in _PREMIUM:
        if not clone_like and max(
            float(race.get("evidence", {}).get(item, 0.0) or 0.0),
            float(race.get("policy_evidence", 0.0) or 0.0),
        ) < _ADAPT_MIN_EVIDENCE:
            continue
        base_price = float(_MARKET_PARAMS[item][0])
        if float(_get(prices, item, 0) or 0) < base_price * _PREEMPT_MIN_PRICE_RATIO:
            continue
        preferred = _adaptive_horizon(obs, step, item)
        # Try the inferred second-order lead first, then back off to the
        # farthest lead for which inventory is actually in the shed.  Horizon
        # one is the exact V17 fallback.
        for horizon in range(preferred, 0, -1):
            future_quantity = _planned_premium(obs, step + horizon, item)
            if future_quantity < _PREEMPT_MIN_FUTURE_QUANTITY:
                continue
            target = min(
                max(0, int(remaining.get(item, 0) or 0)),
                future_quantity,
                _PREEMPT_MAX_BATCH,
                max(1, int(round(future_quantity * _PREEMPT_FRACTION))),
            )
            if target > 0:
                choices.append(
                    (float(_get(prices, item, 0) or 0) * target, item, target, horizon)
                )
                break
    # Preserve V17's behavior before inference.  Once a product is evidence-
    # adapted, shift only the highest-value adapted opportunity this turn.
    adapted = [choice for choice in choices if choice[3] > 1]
    selected = [max(adapted)] if adapted else (choices if clone_like else [])
    if adapted and selected:
        race = _race_state(obs, step)
        race["adapted_shifts"] = int(race.get("adapted_shifts", 0)) + 1
        race["adapted_units"] = int(race.get("adapted_units", 0)) + int(selected[0][2])
    for _, item, target, horizon in selected:
        if len(market) >= 10:
            break
        market.append(["SELL", item, target])
        remaining[item] = max(0, int(remaining.get(item, 0) or 0) - target)
        debts = state.setdefault("debts", {})
        due = debts.setdefault(step + horizon, {})
        due[item] = due.get(item, 0) + target
    if selected:
        action["market"] = market[:10]
    return action


def _tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (_get(farm, "tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError):
        return "LOCKED"


def _trace_actor_action(obs, step, actor):
    actions = _kawa_actions(obs)
    trace = actions[min(max(int(step), 0), len(actions) - 1)] or {}
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
            replayed = _trace_actor_action(obs, step - 1, actor)
            # If a delayed PLANT landed after another actor's same-turn WATER,
            # the crop is still dry.  Use only an inherited idle slot to water
            # it; never displace movement or productive work.
            intended = list(transaction.get("intended") or [])
            tile = _tile_at(farm, positions[index]) if index < len(positions) else None
            if (
                age == 2
                and replayed == ["PASS"]
                and len(intended) >= 2
                and intended[0] == "PLANT"
                and isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and tile.get("crop") == intended[1]
                and not bool(tile.get("watered_today", False))
            ):
                replayed = ["WATER"]
            unit_actions[index] = replayed
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



_V17_R5_MARKETS = json.loads(zlib.decompress(base64.b85decode(
    "c-p;M+is&U5d9aPc>von<S}hoHCozKq*c_7M*IJNu~cDG40E%AN|hRs_<}v>%$Z|fuh;D1<MZ!ZcY6AGe9!Xi^4uKy|D}Wcnmr%8CKEn<H9x!_Uk+{G`tfw>+s+=JpPS|_%iaGk&Q0^wKYnT2(`%ORCXa_H?7oNTKV7qP)3)E=?(x1X-k166V)@@_J}dP%ywtCzdq1|vKTQ|B_jH|S+f>1*FMK1(wk5C)J+KpJyHx#R$wGv?TTULI-@C)*q3OEMuYj0sUBuM5pQ^e^Tl*5$h?vO>bLbk+X1ca1(@YPY#7G!#xnpQ4A_!J^>EuBSth-X^Mss0J04yE}Q{FBs5@rl~CvSW?o!TJ<AZu{X0qx=SX|&m4I2dGIn8EsaD-&W=t~BJzav|WD=-=ZUY59SYsmdzy3%W;fI7<X0I(8+b6Pvb+6z|e08QDEET{PV$?SP|i4M|P213nE8;_6zUzA1~P5aLi83L<U1D1&bp<mK4@9!m=C7$K9?AwMl&lhHG5*?kgEg}S;D*^(eCd>wC{-U5O^Ld_5hQATi?WCpCEm8XZ<F|+eTcV&><r$Y%-z@fW1>(Bv1px-5-goHiCX{F5GYrn9hifb{O>C;Rf-4ug(U?`sCw$h-23djb=Z4oPofDqyrCnZ7srio*dq*M(=D)=zTjnYE0N<!nrK`IK^cA8gHDEje!?qLHZms~zs5_zQsAzQ6UI&}I96hR%d$AW6Y9=X8Uq)NmUf`fFCt%{Uk?F~?xsMgk|oX{~VLL^=&>5zV(c&JTsQiN-Zt5BLMc8E?lp$ZcU3l;N^d%QWzVWQe0(PG>NvWRF$<1{2pw3sB*uT8aH4J`9>!*WBHO|cy9n26;H9GZSHI;-eW>J=5=0$<F)GN*3+o*gXqmSapihqFzc7^gX7G*ht~47}e)7e(cDu4IaL28JK(lf;)L6Jnf7H6#+l)UC)TcrTuYdNGqe@wNp*zdw@mohJ#ef><v=t*HjXfi4P(zGof`I}>W!lGQjvY(5)#2b38yRR{E$E*@t!GMf9DTq4Q}1A=p^VC|{ol*~%G75Sq)f<JiOSIl>|QkF?W{ou5@78f%)Ixkw}*kTLkNufIzot4(w(lbh<$i5VBjD^)VmH^z=iye_cM)HObAM6n<cZwPZChjhaz`?4*Y*c;6TX=kZ#!-P?Nrx>uFdDx)b3!Sno*}jiU?W22@0t0>xHXiBB8=byze2@Z*QkAy2B<J@wozf2&B&mGx;#PD@M`T*A)Uxj5w70E_$!tJt=E%N&Glo1n^=PE3G8huMlegI_~GOr_}w%-qt75L@Q2!X7-|Z?8`F7CC5C5J@(sHPuJ`;(VI@-@f3+!o>&1&9#GwiHj_PYDgRMcQ3c8XkF$C?o0KlYLp)wKfSDu*5C^L`9ud{Ed-W|<Uk^;zC3Q~wY)ceKkvYb=w8iXKu6u+P|tD)`6s8S!Z^Yma0IWGhPHTh5lz5xRsba$8T{m&A*!4L2aQC`i0!$X7wxuFpL0hMf{p8"
)).decode("utf-8"))
_V17_R5_ITEMS = ('MELON', 'MILK', 'STRAWBERRY', 'WOOL')
_V17_R5_FRACTION = 0.5
_V17_R5_STATE = {
    0: {"last_step": -1, "target": False},
    1: {"last_step": -1, "target": False},
}


def _v17_r5_signature(obs):
    seat = _seat(obs)
    farms = list(_get(obs, "farms", []) or [])
    opponent = farms[1 - seat] if len(farms) >= 2 else {}
    cows = sheep = 0
    for row in list(_get(opponent, "tiles", []) or []):
        for tile in list(row or []):
            if not isinstance(tile, dict):
                continue
            cows += int(tile.get("animal") == "COW")
            sheep += int(tile.get("animal") == "SHEEP")
    return cows, sheep


def _v17_is_r5_family(obs, step):
    seat = _seat(obs)
    state = _V17_R5_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "target": False}
        _V17_R5_STATE[seat] = state
    state["last_step"] = step
    if not state.get("target") and step >= 24:
        cows, sheep = _v17_r5_signature(obs)
        if sheep >= 4 and cows <= 3:
            state["target"] = True
    return bool(state.get("target"))


def _v17_town_demand_at(obs, item, step):
    demand = 1 if item != "FERTILIZER" and step % 24 == 0 else 0
    if step % 4 != 0:
        return demand
    town = _get(obs, "town", {}) or {}
    for shop in list(_get(town, "unlocked_shops", []) or []):
        products = _SHOP_PRODUCTS.get(shop, ())
        if item in products:
            demand += 2 if len(products) == 1 else 1
    return demand


def _v17_pickup_reserve(action, item):
    reserve = 0
    orders = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    for order in orders:
        if isinstance(order, (list, tuple)) and len(order) >= 2 and order[0] == "PICKUP" and order[1] == item:
            reserve += max(0, int(order[2])) if len(order) >= 3 else 1
    return reserve


def _v17_r5_counter(obs, action, step):
    if not _v17_is_r5_family(obs, step):
        return action
    future = step + 3
    if future >= len(_V17_R5_MARKETS):
        return action
    targets = {}
    for order in _V17_R5_MARKETS[future]:
        if len(order) >= 3 and order[0] == "SELL" and order[1] in _V17_R5_ITEMS:
            targets[order[1]] = targets.get(order[1], 0) + max(0, int(order[2] or 0))
    if not targets:
        return action
    action = _copy_action(action)
    market = [list(order) for order in action.get("market", []) or []]
    shed = dict(_get(_get(obs, "private", {}) or {}, "shed", {}) or {})
    for item in _V17_R5_ITEMS:
        planned = targets.get(item, 0)
        if planned <= 0:
            continue
        # R5A moves this base sale to step+1 only when town demand does not
        # refill the product before it acts.  Counter only that clean case.
        if _v17_town_demand_at(obs, item, step) > 0 or _v17_town_demand_at(obs, item, step + 1) > 0:
            continue
        existing = sum(
            max(0, int(order[2] or 0))
            for order in market
            if len(order) >= 3 and order[0] == "SELL" and order[1] == item
        )
        available = max(
            0,
            int(shed.get(item, 0) or 0)
            - existing
            - _v17_pickup_reserve(action, item),
        )
        quantity = min(
            available,
            max(1, int(round(planned * _V17_R5_FRACTION))),
        )
        if quantity <= 0:
            continue
        current = next(
            (order for order in market if len(order) >= 3 and order[0] == "SELL" and order[1] == item),
            None,
        )
        if current is not None:
            current[2] = max(0, int(current[2] or 0)) + quantity
        elif len(market) < 10:
            market.append(["SELL", item, quantity])
        else:
            continue
    action["market"] = market[:10]
    return action


def _shape(name, value, scale=None):
    value = max(0.0, float(value))
    if name == "hinge":
        if scale is None or float(scale) <= 0:
            raise ValueError("hinge requires a positive scale")
        u = value / float(scale)
        return u + 8.0 * max(0.0, u - 1.0) ** 2
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



_V17_MD_MARKETS = json.loads(zlib.decompress(base64.b85decode(
    "c-q}u&2HN;41O1%eF$a8KgYE7&|qm(xE+eF5cd9Wv35zD*d`^CqMcws4}~q$6h(ggNK1Ktf6wl>eV6&1_s`9*w?CW5?Zal5<=O52HOt-P^7DPyJ)PZn?z+2=%dhv{<|WJP(dCD3w|~rX_#Xb$@9%!yzMP(@{Ku{L?77?RP8W;Mi|2pD!)`o|9ty}%tG{pce{}uJcDMcA^`CQ~ro2YAjxLauC7d^zU&-~VZ|yM$gOS7BZu)+2w_FyQ)1Hfl!1^@R;SD$Al-8fR3}dLlBN`-gK2G5IrQf{XbbbGJoCWzV^Z^_aFgc;IJln`Up0;O#m5Sh`gEKtYWWV2?dD(9Bc$eW~osZ9*5=%-N1!V0Lt;~1^U5ZMWnLriY!%!`K5R&Z?mS=@@%z_m@;bBxiY<E8~!&_MRJW5Jj892ATz_Y*9PFk3NLc|UB@(r=B4EVJ$ty1hmD2B?|6EhYhT9ux-Mhh#W;grG#dDJRs2i|a6t=jX${3AZ;l%#kF8qDz|qpJ{>$ON%Q4=M(02|!O~XkJW4Y=TZCp^9M&E{TNh>5U?)4~6r?1p7@qEFj+%-XJ7p!4|f15boIt6nt_9VI>%B?b7Ljd`V~vH9i7yK6r<pju8npJE+c|Y69|dl>&6U&vNIGXB#}S@q~ewiwh-6c6reH*~r?+RIOnkA!1xSoT3e^mBLS0gcqfcnmf5_JpxX<v~OY+9%a>$+GL_{)p$soy7h7_YZTJJ(YGooHbN!hiy)S7F65lJQ~_!%6Rw}rKbWU@Aoyqpg<11sdK83N+(OYRZ!NXVO6o|A$(j399*GLgXMUhGdcfJeVU)lSyN7+Sow{0mxx_%jY#t)URFq3Vt&qUQ8nc?@ZS_PX{bdUvQ8dV^@D<b8kr=NPeBPGnLurzcoRy2Mg`*iz5W`s)Sj$k8KHA@aiZEcK=k^E5jY*T0k;@tdM=ZUZaXQ6&#B{`YO>02-5>tK6Cb*W0QalhQbrb*ReN2WctH4?n(3M0AlBeq&`<xoevPi~5{aZj|TTx~&MWaD9dX>&G$kDYue@P}O=`6>fw5_FtT|y*k3su0%2ehv|z@KXw`1fT!cL|^icKrDJUgF=T(S}>iH&1gN^;YzIpp&?ls6uLtQV3;q5lpv6%2czxv^8jWsZA2aIL(*s(gOb6HDjn%2B9`~N?*FRFjlp!kl}PjAHW$c)=T!cctH``zFlG$1A8xu@CVogUhNhdJZX$_<Fa0!d)4XuI|Zr^q@z^NdrDe_b~{n<Sr8b9JvSrex6u%ivca*{wvf(olm!Z95yxw<k@NG}r>uyV_Zg6N7V&G4aW4^!w3k=pfG09VoJf%(85qafx`bMQ;(dD8gYwdif&|@Z41z|md6Z@~!eXw~u3wbznALcE8eP!FEr|UX>1QMwY<G{IiN|-hvFNk&+^r7LR<mZ$M8R}fr*K)vlnt7Ko!gl&xWcC}OWG;Yz7UjI=xMn%c~}d`xM>3WRU8W^28{NyCo38+I}$1E0V=3c#os0+tO-qcTZJ$IE8fD+uSH_%=Ip-+IrZ~{oP?`7*Y?9}{pw=<N*c=Yxr$z^ih8+~h)jq{q@qWh!lmHptnkHC!?cc`Ce0ZrO6c$@fS#mis90_Dy2{|gfdzU+l|C}{l~vwn_rX9}&&2mo$amNpOlM&sG*@jo$CnPPj>H5s*N;Nv`K?e~m)uSfaxg_#y0wVdLWYQ@AB_2^$Lg+dt01*gtNSo9?NmxIkV#=S%=HFc_<<!`R^xCI7K7Ro*@JwMAXof=0lB3G{C{)tY>PDWI2CXXeoT5QUp`SSbJ~b`hBWq*6f~AjCK%}>pzSFj2Kv8<5=ij"
)).decode("utf-8"))
_V17_MD_FRACTION = 2.0
_V17_ROOM_GUARD = True
_V17_FEED_GUARD = False
_V17_MD_ITEMS = ("MELON", "MILK", "STRAWBERRY", "WOOL")
_V17_MD_STATE = {
    0: {"last_step": -1, "target": False},
    1: {"last_step": -1, "target": False},
}
_V17_FEED_RESCUE_STATE = {
    0: {"last_step": -1, "day": -1, "active": {}},
    1: {"last_step": -1, "day": -1, "active": {}},
}
_V17_ROOM_EVAC_STATE = {
    0: {"last_step": -1, "day": -1, "active": None},
    1: {"last_step": -1, "day": -1, "active": None},
}


def _v17_md_signature(obs):
    seat = _seat(obs)
    farms = list(_get(obs, "farms", []) or [])
    opponent = farms[1 - seat] if len(farms) >= 2 else {}
    cows = sheep = 0
    for row in list(_get(opponent, "tiles", []) or []):
        for tile in list(row or []):
            if not isinstance(tile, dict):
                continue
            cows += int(tile.get("animal") == "COW")
            sheep += int(tile.get("animal") == "SHEEP")
    quadrants = len(_get(opponent, "unlocked_quadrants", []) or [])
    return cows, sheep, quadrants


def _v17_is_md_family(obs, step):
    seat = _seat(obs)
    state = _V17_MD_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "target": False}
        _V17_MD_STATE[seat] = state
    state["last_step"] = step
    if not state.get("target") and step >= 160:
        cows, sheep, quadrants = _v17_md_signature(obs)
        if (quadrants >= 2 and cows >= 4 and sheep <= 2) or cows >= 9:
            state["target"] = True
    return bool(state.get("target"))


def _v17_md_pickup_reserve(action, item):
    reserve = 0
    for order in [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]:
        if isinstance(order, (list, tuple)) and len(order) >= 2 and order[0] == "PICKUP" and order[1] == item:
            reserve += max(0, int(order[2])) if len(order) >= 3 else 1
    return reserve


def _v17_md_counter(obs, action, step):
    if _V17_MD_FRACTION <= 0 or not _v17_is_md_family(obs, step) or step + 1 >= len(_V17_MD_MARKETS):
        return action
    targets = {}
    for order in _V17_MD_MARKETS[step + 1]:
        if len(order) >= 3 and order[0] == "SELL" and order[1] in _V17_MD_ITEMS:
            targets[order[1]] = targets.get(order[1], 0) + max(0, int(order[2] or 0))
    if not targets:
        return action
    action = _copy_action(action)
    market = [list(order) for order in (action.get("market") or [])]
    shed = dict(_get(_get(obs, "private", {}) or {}, "shed", {}) or {})
    for item in _V17_MD_ITEMS:
        target = targets.get(item, 0)
        if target <= 0:
            continue
        existing_quantity = sum(
            max(0, int(order[2] or 0))
            for order in market
            if len(order) >= 3 and order[0] == "SELL" and order[1] == item
        )
        available = max(
            0,
            int(shed.get(item, 0) or 0)
            - existing_quantity
            - _v17_md_pickup_reserve(action, item),
        )
        quantity = min(available, max(1, int(round(target * _V17_MD_FRACTION))))
        if quantity <= 0:
            continue
        existing = next(
            (order for order in market if len(order) >= 3 and order[0] == "SELL" and order[1] == item),
            None,
        )
        if existing is not None:
            existing[2] = max(0, int(existing[2] or 0)) + quantity
        elif len(market) < 10:
            market.append(["SELL", item, quantity])
        else:
            continue
    action["market"] = market[:10]
    return action


def _v17_move_toward(position, target):
    x, y = int(position[0]), int(position[1])
    tx, ty = int(target[0]), int(target[1])
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def _v17_feed_guard(obs, action, step):
    hour = int(_get(obs, "hour", 0) or 0)
    day = int(_get(obs, "day", step // 24) or 0)
    if not _V17_FEED_GUARD or hour < 18:
        return action
    action = _align_hands(action, obs)
    seat = _seat(obs)
    state = _V17_FEED_RESCUE_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)) or day != int(state.get("day", -1)):
        state = {"last_step": step, "day": day, "active": {}}
        _V17_FEED_RESCUE_STATE[seat] = state
    state["last_step"] = step
    farm = _farm(obs, seat)
    private = _get(obs, "private", {}) or {}
    positions = [_get(farm, "farmer", [4, 4]), *list(_get(farm, "hands", []) or [])]
    inventories = list(_get(private, "inventories", []) or [])
    orders = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]

    threats = []
    for y, row in enumerate(list(_get(farm, "tiles", []) or [])):
        for x, tile in enumerate(list(row or [])):
            if (
                isinstance(tile, dict)
                and tile.get("animal")
                and int(tile.get("consecutive_unfed", 0) or 0) >= 1
                and not tile.get("fed_today", False)
            ):
                threats.append((x, y))
    threat_set = set(threats)
    active = state.setdefault("active", {})
    for actor, target in list(active.items()):
        actor = int(actor)
        if actor >= len(positions) or actor >= len(inventories) or tuple(target) not in threat_set:
            active.pop(actor, None)
            continue
        inventory = dict(inventories[actor] or {})
        if int(inventory.get("WHEAT", 0) or 0) <= 0:
            active.pop(actor, None)
            continue
        if tuple(positions[actor]) == tuple(target):
            orders[actor] = ["FEED"]
        else:
            orders[actor] = _v17_move_toward(positions[actor], target)

    claimed = {tuple(target) for target in active.values()}
    remaining_actions = max(1, 24 - hour)
    for target in threats:
        if target in claimed:
            continue
        if any(
            tuple(position) == target
            and actor < len(orders)
            and orders[actor]
            and orders[actor][0] == "FEED"
            for actor, position in enumerate(positions)
        ):
            continue
        candidates = []
        for actor, position in enumerate(positions):
            if actor in active or actor >= len(inventories):
                continue
            if int(dict(inventories[actor] or {}).get("WHEAT", 0) or 0) <= 0:
                continue
            distance = abs(int(position[0]) - target[0]) + abs(int(position[1]) - target[1])
            if distance + 1 <= remaining_actions:
                candidates.append((distance, actor))
        if not candidates:
            continue
        distance, actor = min(candidates)
        # Do not seize a worker early; start only at the last safe moment.
        if distance + 1 < remaining_actions:
            continue
        active[actor] = list(target)
        claimed.add(target)
        orders[actor] = ["FEED"] if distance == 0 else _v17_move_toward(positions[actor], target)
    action["farmer"] = orders[0] if orders else ["PASS"]
    action["hands"] = orders[1:]
    return action


def _v17_room_evac(obs, action, step):
    if not _V17_ROOM_GUARD or step < 648:
        return action
    hour = int(_get(obs, "hour", 0) or 0)
    day = int(_get(obs, "day", step // 24) or 0)
    seat = _seat(obs)
    state = _V17_ROOM_EVAC_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)) or day != int(state.get("day", -1)):
        state = {"last_step": step, "day": day, "active": None}
        _V17_ROOM_EVAC_STATE[seat] = state
    state["last_step"] = step
    if hour < 21:
        return action
    action = _align_hands(action, obs)
    farm = _farm(obs, seat)
    private = _get(obs, "private", {}) or {}
    positions = [_get(farm, "farmer", [4, 4]), *list(_get(farm, "hands", []) or [])]
    inventories = [dict(value or {}) for value in list(_get(private, "inventories", []) or [])]
    orders = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    shed = dict(_get(private, "shed", {}) or {})
    total = sum(max(0, int(value or 0)) for value in shed.values()) + sum(
        max(0, int(value or 0)) for inventory in inventories for value in inventory.values()
    )
    access = _shed_access(len(_get(farm, "tiles", []) or []) or 10)
    if hour == 21 and state.get("active") is None and total > 100:
        candidates = []
        for actor, (position, inventory) in enumerate(zip(positions, inventories)):
            saleable = sum(max(0, int(inventory.get(item, 0) or 0)) for item in _SELLABLE)
            if saleable <= 0 or actor >= len(orders) or (orders[actor] and orders[actor][0] != "PASS"):
                continue
            target = min(access, key=lambda point: abs(int(position[0]) - point[0]) + abs(int(position[1]) - point[1]))
            distance = abs(int(position[0]) - target[0]) + abs(int(position[1]) - target[1])
            if distance <= 2:
                candidates.append((distance, -saleable, actor, target))
        if candidates:
            _, _, actor, target = min(candidates)
            state["active"] = {"actor": actor, "target": list(target)}
    active = state.get("active")
    if active is None:
        return action
    actor = int(active["actor"])
    target = tuple(active["target"])
    if actor >= len(positions) or actor >= len(inventories):
        state["active"] = None
        return action
    if tuple(positions[actor]) != target:
        orders[actor] = _v17_move_toward(positions[actor], target)
    elif hour == 23:
        orders[actor] = ["DROP"]
        market = [list(order) for order in (action.get("market") or [])]
        existing_sales = {}
        for order in market:
            if len(order) >= 3 and order[0] == "SELL":
                existing_sales[order[1]] = existing_sales.get(order[1], 0) + max(0, int(order[2] or 0))
        needed = max(0, total - 100)
        priority = ("WOOL", "MILK", "EGG", "MELON", "STRAWBERRY", "TOMATO", "CARROT", "FERTILIZER", "WHEAT")
        inventory = inventories[actor]
        for item in priority:
            available = max(0, int(inventory.get(item, 0) or 0) - existing_sales.get(item, 0))
            quantity = min(needed, available)
            if quantity <= 0:
                continue
            existing = next(
                (order for order in market if len(order) >= 3 and order[0] == "SELL" and order[1] == item),
                None,
            )
            if existing is not None:
                existing[2] = int(existing[2] or 0) + quantity
            elif len(market) < 10:
                market.append(["SELL", item, quantity])
            else:
                continue
            needed -= quantity
            if needed <= 0:
                break
        action["market"] = market[:10]
    action["farmer"] = orders[0] if orders else ["PASS"]
    action["hands"] = orders[1:]
    return action


def _v17_room_guard(obs, action, step):
    if not _V17_ROOM_GUARD or step % 24 != 23:
        return action
    action = _copy_action(action)
    private = _get(obs, "private", {}) or {}
    shed = {key: max(0, int(value or 0)) for key, value in dict(_get(private, "shed", {}) or {}).items()}
    inventories = [dict(value or {}) for value in list(_get(private, "inventories", []) or [])]
    carried = sum(max(0, int(value or 0)) for inventory in inventories for value in inventory.values())
    farm = _farm(obs, _seat(obs))
    positions = [_get(farm, "farmer", [4, 4]), *list(_get(farm, "hands", []) or [])]
    orders = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    produced = consumed = 0
    for actor, order in enumerate(orders):
        if actor >= len(positions) or not isinstance(order, list) or not order:
            continue
        tile = _tile_at(farm, positions[actor])
        if order[0] == "HARVEST" and isinstance(tile, dict):
            produced += max(0, int(tile.get("yield_units", 0) or 0))
        elif order[0] == "COLLECT_FERTILIZER" and isinstance(tile, dict) and tile.get("fertilizer_available", False):
            produced += 1
        elif order[0] in ("FEED", "FERTILIZE"):
            consumed += 1
        elif order[0] == "PLACE" and len(order) >= 2 and order[1] in ("GOOSE", "COW", "SHEEP"):
            consumed += 1
    market = [list(order) for order in (action.get("market") or [])]
    planned_sells = {}
    planned_buys = 0
    for order in market:
        if len(order) < 3:
            continue
        quantity = max(0, int(order[2] or 0))
        if order[0] == "SELL":
            planned_sells[order[1]] = planned_sells.get(order[1], 0) + quantity
        elif order[0] in ("BUY_PRODUCT", "BUY_ANIMAL"):
            planned_buys += quantity
    actual_existing_sells = sum(min(shed.get(item, 0), quantity) for item, quantity in planned_sells.items())
    needed = max(
        0,
        sum(shed.values()) + carried + produced - consumed + planned_buys - actual_existing_sells - 100,
    )
    if needed <= 0:
        return action
    # Finished animal products and sale-only crops are safest to liquidate.
    priority = ("WOOL", "MILK", "EGG", "MELON", "STRAWBERRY", "TOMATO", "CARROT", "FERTILIZER", "WHEAT")
    for item in priority:
        already = planned_sells.get(item, 0)
        available = max(0, shed.get(item, 0) - already)
        quantity = min(needed, available)
        if quantity <= 0:
            continue
        existing = next(
            (order for order in market if len(order) >= 3 and order[0] == "SELL" and order[1] == item),
            None,
        )
        if existing is not None:
            existing[2] = int(existing[2] or 0) + quantity
        elif len(market) < 10:
            market.append(["SELL", item, quantity])
        else:
            continue
        planned_sells[item] = already + quantity
        needed -= quantity
        if needed <= 0:
            break
    action["market"] = market[:10]
    return action


def _v20_move_toward(position, target, tiles):
    x, y = int(position[0]), int(position[1])
    tx, ty = int(target[0]), int(target[1])
    choices = []
    if tx < x:
        choices.append(("WEST", (x - 1, y)))
    if tx > x:
        choices.append(("EAST", (x + 1, y)))
    if ty < y:
        choices.append(("NORTH", (x, y - 1)))
    if ty > y:
        choices.append(("SOUTH", (x, y + 1)))
    size = len(tiles)
    for operation, (nx, ny) in choices:
        if 0 <= nx < size and 0 <= ny < size and tiles[ny][nx] != "LOCKED":
            return [operation]
    return ["PASS"]


def _v20_terminal_action(obs):
    seat = _seat(obs)
    farm = _farm(obs, seat)
    private = _get(obs, "private", {}) or {}
    tiles = list(_get(farm, "tiles", []) or [])
    size = len(tiles)
    positions = [_get(farm, "farmer", [0, 0]), *list(_get(farm, "hands", []) or [])]
    inventories = list(_get(private, "inventories", []) or [])
    inventories.extend({} for _ in range(len(positions) - len(inventories)))
    sheds = set(_shed_access(size))
    available = {
        (x, y)
        for y, row in enumerate(tiles)
        for x, tile in enumerate(row)
        if isinstance(tile, dict) and int(tile.get("yield_units", 0) or 0) > 0
    }
    actions = []
    pending = {}
    for raw_position, inventory in zip(positions, inventories):
        position = tuple(raw_position)
        inventory = inventory or {}
        load = sum(max(0, int(value or 0)) for value in inventory.values())
        x, y = position
        tile = tiles[y][x] if 0 <= y < size and 0 <= x < size else None
        if load > 0 and position in sheds:
            unit_action = ["DROP"]
            for item, count in inventory.items():
                if item in _SELLABLE:
                    pending[item] = pending.get(item, 0) + max(0, int(count or 0))
        elif isinstance(tile, dict) and int(tile.get("yield_units", 0) or 0) > 0:
            unit_action = ["HARVEST"]
            available.discard(position)
        elif load > 0:
            target = min(sheds, key=lambda cell: abs(cell[0] - x) + abs(cell[1] - y))
            unit_action = _v20_move_toward(position, target, tiles)
        elif available:
            target = min(
                available,
                key=lambda cell: (abs(cell[0] - x) + abs(cell[1] - y), cell[1], cell[0]),
            )
            available.discard(target)
            unit_action = _v20_move_toward(position, target, tiles)
        elif isinstance(tile, dict) and tile.get("fertilizer_available", False):
            unit_action = ["COLLECT_FERTILIZER"]
        else:
            unit_action = ["PASS"]
        actions.append(unit_action)
    shed = dict(_get(private, "shed", {}) or {})
    for item, count in pending.items():
        shed[item] = int(shed.get(item, 0) or 0) + count
    prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
    sells = [
        (int(shed.get(item, 0) or 0) * int(_get(prices, item, 1) or 1), item,
         int(shed.get(item, 0) or 0))
        for item in _SELLABLE
        if int(shed.get(item, 0) or 0) > 0
    ]
    sells.sort(reverse=True)
    return {
        "farmer": actions[0] if actions else ["PASS"],
        "hands": actions[1:],
        "market": [["SELL", item, quantity] for _, item, quantity in sells[:10]],
    }


_V38_TOMATO_TARGET = 3
_V38_TOMATO_STATE = {
    0: {"last_step": -1, "active": False, "scheduled_plants": 0},
    1: {"last_step": -1, "active": False, "scheduled_plants": 0},
}


def _v38_farm_pair_tomatoes(obs, action, step):
    """Convert a prefix-aligned part of the day-11 strawberry cohort."""
    seat = _seat(obs)
    state = _V38_TOMATO_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "active": False, "scheduled_plants": 0}
        _V38_TOMATO_STATE[seat] = state
    state["last_step"] = step
    if step == 216:
        shops = list(_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or [])[:3]
        state["active"] = sum(
            shop in ("FARMERS_MARKET", "PIZZA_SHOP") for shop in shops
        ) >= 2
    if not state.get("active"):
        return action

    action = _copy_action(action)
    market = list(action.get("market") or [])
    if step == 264:
        strawberry_buy = next(
            (
                order
                for order in market
                if isinstance(order, list)
                and len(order) >= 3
                and order[:2] == ["BUY_SEED", "STRAWBERRY"]
                and int(order[2] or 0) >= _V38_TOMATO_TARGET
            ),
            None,
        )
        if strawberry_buy is not None:
            strawberry_buy[2] = int(strawberry_buy[2] or 0) - _V38_TOMATO_TARGET
            state["seed_debt"] = _V38_TOMATO_TARGET
    if step == 265 and int(state.get("seed_debt", 0) or 0) > 0 and len(market) < 10:
        market.append(["BUY_SEED", "TOMATO", int(state["seed_debt"])])
        state["seed_debt"] = 0

    private = _get(obs, "private", {}) or {}
    inventories = [dict(value or {}) for value in list(_get(private, "inventories", []) or [])]
    orders = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    if 271 <= step <= 286:
        remaining = max(0, _V38_TOMATO_TARGET - int(state.get("scheduled_plants", 0)))
        changed = 0
        for order in orders:
            if (
                changed < remaining
                and isinstance(order, list)
                and len(order) >= 2
                and order[:2] == ["PLANT", "STRAWBERRY"]
            ):
                order[1] = "TOMATO"
                changed += 1
        state["scheduled_plants"] = int(state.get("scheduled_plants", 0)) + changed
    for actor, order in enumerate(orders):
        if (
            actor < len(inventories)
            and isinstance(order, list)
            and len(order) >= 2
            and order[:2] == ["PLACE", "STRAWBERRY"]
            and int(inventories[actor].get("TOMATO", 0) or 0) > 0
        ):
            order[1] = "TOMATO"
    action["farmer"] = orders[0]
    action["hands"] = orders[1:]

    if step >= 432 and step % 24 == 0:
        shed = dict(_get(private, "shed", {}) or {})
        tomatoes = max(0, int(shed.get("TOMATO", 0) or 0))
        existing = next(
            (
                order
                for order in market
                if isinstance(order, list)
                and len(order) >= 3
                and order[:2] == ["SELL", "TOMATO"]
            ),
            None,
        )
        if tomatoes and existing is not None:
            existing[2] = max(int(existing[2] or 0), tomatoes)
        elif tomatoes and len(market) < 10:
            market.append(["SELL", "TOMATO", tomatoes])
    # Let the scarcity patch compound before monetizing this small cohort.
    # The terminal controller at step 708 sees the held shed stock and liquidates it.
    market = [
        order
        for order in market
        if not (
            isinstance(order, list)
            and len(order) >= 2
            and order[:2] == ["SELL", "TOMATO"]
        )
    ]
    action["market"] = market[:10]
    return action


# Ryo's public route begins relaying carrots after its fifth shop, while
# Subramanya's largest public carrot cohort uses the ordinary crop relays.
# V58 has both physical seams.  Safe early wheat plots receive one mature
# watering and a later inherited WATER visit before carrot expiry; that later
# visit can harvest the carrot without changing movement.  The step-613..622
# terminal wheat cohort already has inherited water and harvest orders.  Both
# replacements therefore need no new worker, movement, or route overlay.
_V71_CARROT_TARGET = 12
_V71_CARROT_START = 613
_V71_CARROT_STOP = 622
_V71_EARLY_EXPECTED_UNITS = 0
_V71_EARLY_MIN_STEP = 589
_V71_EARLY_MAX_STEP = 610
_V71_EARLY_QUOTE_GATE = 110
# These are inherited V58 plant slots whose routes revisit the same plot for
# one mature watering and then again before carrot max_lifespan_step.  The set
# covers both public layout branches; a slot is used only when the live action
# is actually PLANT WHEAT.
_V71_EARLY_SLOTS = {
    # Normal layout, first relay.
    (541, 7), (545, 10), (546, 11), (547, 0),
    (547, 4), (548, 2), (549, 7), (549, 11),
    # Legacy/public Wufang layout, first relay.
    (541, 10), (542, 7), (543, 11), (545, 12),
    # Normal layout, second relay.
    (569, 7), (569, 10), (571, 5), (573, 6),
    # Legacy/public Wufang layout, second relay.
    (565, 7), (570, 7), (571, 1), (572, 0), (572, 5),
    (572, 9), (573, 3), (574, 8),
    # Normal layout, third relay.
    (589, 9), (594, 11), (595, 2), (595, 8), (595, 9),
    (597, 6), (598, 2), (598, 9),
    # Legacy/public Wufang layout, third relay.
    (594, 7), (598, 11), (610, 11),
}
_V71_POST_EXPECTED_UNITS = 36
# Final inherited wheat plants whose ordinary harvest lands no later than
# step 707.  Carrot and wheat both realize two units on these routes, but the
# former is worth materially more under the confirmed scarcity gate.  Crops
# with a source harvest at 708+ are deliberately excluded because V58's
# terminal controller takes over then.
_V71_POST_SLOTS = {
    # Normal layout.
    (636, 11), (640, 8), (641, 7), (643, 0), (643, 4),
    (643, 5), (643, 8), (645, 2), (646, 0), (655, 7),
    (659, 10), (665, 11), (666, 0), (666, 6), (668, 3),
    (669, 0), (669, 4), (670, 10),
    # Legacy/public Wufang layout.
    (639, 2), (641, 1), (642, 11), (645, 0), (645, 10),
    (664, 2), (664, 5), (665, 6), (666, 0), (666, 11),
    (667, 1), (668, 4), (668, 5), (670, 2), (670, 6),
}
_V71_BASE_CARROT_UNITS = 11
_V71_EXPECTED_YIELD = 3
_V71_CARROT_STATE = {
    0: {
        "last_step": -1,
        "preliminary": False,
        "active": False,
        "early_active": False,
        "decision_quote": 0,
        "plants": 0,
        "post_plants": 0,
        "seed_debt": 0,
        "lost_wheat": 0,
        "early_tiles": {},
    },
    1: {
        "last_step": -1,
        "preliminary": False,
        "active": False,
        "early_active": False,
        "decision_quote": 0,
        "plants": 0,
        "post_plants": 0,
        "seed_debt": 0,
        "lost_wheat": 0,
        "early_tiles": {},
    },
}


def _v71_live_units(farm, crop):
    units = 0
    for row in list(_get(farm, "tiles", []) or []):
        for tile in list(row or []):
            if (
                isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and tile.get("crop") == crop
            ):
                units += max(1, int(tile.get("yield_units", 0) or 0))
    return units


def _v71_animal_count(farm):
    return sum(
        1
        for row in list(_get(farm, "tiles", []) or [])
        for tile in list(row or [])
        if isinstance(tile, dict) and tile.get("animal")
    )


def _v71_carrot_decision(obs, target_units):
    """Public demand/deficit gate with a conservative replacement NPV."""
    step = int(_get(obs, "step", 0) or 0)
    shops = list(_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or [])
    shops = shops[: max(3, len(shops))]
    carrot_shop_score = sum(
        2 if shop == "PET_CAFE" else 1 if shop == "FARMERS_MARKET" else 0
        for shop in shops
    )
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    prices = _get(market, "prices", {}) or {}
    carrot_inventory = int(_get(inventory, "CARROT", 10000) or 10000)
    remaining_demand = sum(
        _town_drain(future, shops, "CARROT")
        for future in range(step, 720)
    )
    # The eight shops unlock deterministically every three days, but their
    # identities remain random.  A future shop consumes an expected 3/8 of a
    # carrot per town tick: PET_CAFE contributes 2/8 and FARMERS_MARKET 1/8.
    # This is distributional demand, not a seed or shop-sequence lookup.
    for shop_number in range(len(shops) + 1, 9):
        unlock_step = 72 * shop_number
        ticks = sum(
            1
            for future in range(max(step, unlock_step), 720)
            if future % 4 == 0
        )
        remaining_demand += (3.0 / 8.0) * ticks
    seat = _seat(obs)
    farms = list(_get(obs, "farms", []) or [])
    opponent = farms[1 - seat] if len(farms) >= 2 else {}
    # A visible live opponent carrot is valued at its full four-unit capacity.
    opponent_carrots = sum(
        1
        for row in list(_get(opponent, "tiles", []) or [])
        for tile in list(row or [])
        if (
            isinstance(tile, dict)
            and tile.get("kind") == "PLANT"
            and tile.get("crop") == "CARROT"
        )
    )
    opponent_capacity = 4 * opponent_carrots
    opponent_strawberries = sum(
        1
        for row in list(_get(opponent, "tiles", []) or [])
        for tile in list(row or [])
        if (
            isinstance(tile, dict)
            and tile.get("kind") == "PLANT"
            and tile.get("crop") == "STRAWBERRY"
        )
    )
    opponent_animals = _v71_animal_count(opponent)
    # At the confirmation step, preserve the incumbent route against visibly
    # lower-throughput animal-heavy farms.  A live carrot cohort or a dense
    # strawberry surface is direct evidence that the rival can contest the
    # same late crop route; compact <=12-animal layouts are the other public
    # high-throughput shape in the replay set.
    rival_route_ready = bool(
        step < 288
        or opponent_carrots > 0
        or opponent_strawberries >= 36
        or opponent_animals <= 12
    )
    projected_inventory = (
        carrot_inventory
        - remaining_demand
        + opponent_capacity
        + _V71_BASE_CARROT_UNITS
        + max(0, int(target_units))
    )
    carrot_quote = _market_price("CARROT", projected_inventory)
    _V71_CARROT_STATE[seat]["decision_quote"] = carrot_quote
    wheat_quote = int(_get(prices, "WHEAT", 25) or 25)
    replacement_npv = (
        _V71_EXPECTED_YIELD * (carrot_quote - wheat_quote)
        - 10  # carrot seed costs ten more than the replaced wheat seed
    )
    return bool(
        carrot_shop_score >= 3
        and rival_route_ready
        and carrot_inventory < 9975
        and projected_inventory <= 9575
        and replacement_npv >= 20
    )


def _v71_pair_seed_debt(action, state):
    """Replace matching wheat seed replenishment without adding order pressure."""
    debt = max(0, int(state.get("seed_debt", 0) or 0))
    if debt <= 0:
        return action
    market = list(action.get("market") or [])
    for order in market:
        if debt <= 0:
            break
        if not (
            isinstance(order, list)
            and len(order) >= 3
            and order[:2] == ["BUY_SEED", "WHEAT"]
        ):
            continue
        quantity = max(0, int(order[2] or 0))
        # Cohort steps buy exactly as many wheat seeds as they plant.  Only an
        # exact prefix is relabeled, so this stays one market order.
        transfer = min(debt, quantity)
        if transfer == quantity:
            order[1] = "CARROT"
            debt -= transfer
        elif len(market) < 10:
            order[2] = quantity - transfer
            market.append(["BUY_SEED", "CARROT", transfer])
            debt -= transfer
    state["seed_debt"] = debt
    action["market"] = market[:10]
    return action


def _v71_route_carrots(obs, action, step):
    """Relabel complete inherited wheat cohorts after two public decisions."""
    seat = _seat(obs)
    state = _V71_CARROT_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {
            "last_step": step,
            "preliminary": False,
            "active": False,
            "early_active": False,
            "decision_quote": 0,
            "plants": 0,
            "post_plants": 0,
            "seed_debt": 0,
            "lost_wheat": 0,
            "early_tiles": {},
        }
        _V71_CARROT_STATE[seat] = state
    state["last_step"] = step

    target_units = (
        _V71_CARROT_TARGET * _V71_EXPECTED_YIELD
        + _V71_EARLY_EXPECTED_UNITS
        + _V71_POST_EXPECTED_UNITS
    )
    if step == 216:
        state["preliminary"] = _v71_carrot_decision(obs, target_units)
    if step == 288:
        state["active"] = bool(
            state.get("preliminary")
            and _v71_carrot_decision(obs, target_units)
        )
        # A small third relay is worthwhile only in the genuinely scarce
        # tail.  This uses the same observation-derived projected quote as the
        # main NPV gate; it is not tied to an opponent, seed, or shop tape.
        state["early_active"] = bool(
            state.get("active")
            and int(state.get("decision_quote", 0) or 0)
            >= _V71_EARLY_QUOTE_GATE
        )
    if not state.get("active"):
        return action

    action = _copy_action(action)
    private = _get(obs, "private", {}) or {}
    farm = _farm(obs, seat)
    inventories = list(_get(private, "inventories", []) or [])
    wheat_stock = int(_get(_get(private, "shed", {}) or {}, "WHEAT", 0) or 0)
    wheat_stock += sum(
        max(0, int(_get(inventory, "WHEAT", 0) or 0))
        for inventory in inventories
    )
    live_wheat = _v71_live_units(farm, "WHEAT")
    feed_reserve = 2 * _v71_animal_count(farm)
    # live_wheat is read from the *current* board, so previously converted
    # plots have already disappeared from it.  Subtracting lost_wheat again
    # would double-count the replacement and unnecessarily clip later plants.
    available_wheat = wheat_stock + live_wheat

    orders = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    positions = [
        _get(farm, "farmer", [4, 4]),
        *list(_get(farm, "hands", []) or []),
    ]
    early_tiles = state.setdefault("early_tiles", {})

    # Drop a tracked plot only after the PLANT action has had a turn to become
    # visible.  This also makes failed plant attempts self-healing instead of
    # causing unrelated future route visits to be rewritten.
    for position, record in list(early_tiles.items()):
        tile = _tile_at(farm, position)
        if (
            step > int(record.get("plant_step", step) or step) + 1
            and not (
                isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and tile.get("crop") == "CARROT"
            )
        ):
            early_tiles.pop(position, None)

    # The whitelist is a mechanical property of the inherited route, not an
    # opponent or seed lookup.  A slot is converted only when the live action
    # remains PLANT WHEAT and the visible wheat supply still covers two units
    # of feed per animal.
    early_changed = 0
    for actor, (order, position) in enumerate(zip(orders, positions)):
        if not state.get("early_active"):
            continue
        if not (_V71_EARLY_MIN_STEP <= step <= _V71_EARLY_MAX_STEP):
            continue
        if (step, actor) not in _V71_EARLY_SLOTS:
            continue
        if available_wheat - _V71_EXPECTED_YIELD < feed_reserve:
            break
        if not (
            isinstance(order, list)
            and len(order) >= 2
            and order[:2] == ["PLANT", "WHEAT"]
        ):
            continue
        order[1] = "CARROT"
        key = (int(position[0]), int(position[1]))
        early_tiles[key] = {
            "plant_step": step,
            "mature_watered": False,
        }
        early_changed += 1
        available_wheat -= _V71_EXPECTED_YIELD

    if early_changed:
        state["seed_debt"] = int(state.get("seed_debt", 0) or 0) + early_changed
        state["lost_wheat"] = int(state.get("lost_wheat", 0) or 0) + (
            early_changed * _V71_EXPECTED_YIELD
        )

    # Carrots mature one day sooner than the inherited wheat.  Keep the first
    # mature WATER (raising the crop to two units at end of day), then turn the
    # next already-routed WATER on that tile into HARVEST.  All whitelisted
    # plots have this second visit before max_lifespan_step.
    day = int(_get(obs, "day", step // 24) or 0)
    for actor, (order, position) in enumerate(zip(orders, positions)):
        key = (int(position[0]), int(position[1]))
        record = early_tiles.get(key)
        if record is None or not isinstance(order, list) or not order:
            continue
        tile = _tile_at(farm, position)
        if not (
            isinstance(tile, dict)
            and tile.get("kind") == "PLANT"
            and tile.get("crop") == "CARROT"
        ):
            continue
        if order[0] == "HARVEST":
            early_tiles.pop(key, None)
            continue
        if order[0] != "WATER":
            continue
        planted_day = int(tile.get("planted_day", day) or day)
        if day < planted_day + 2:
            continue
        if record.get("mature_watered"):
            order[:] = ["HARVEST"]
            early_tiles.pop(key, None)
        else:
            record["mature_watered"] = True

    if early_changed:
        action["farmer"] = orders[0]
        action["hands"] = orders[1:]

    if _V71_CARROT_START <= step <= _V71_CARROT_STOP:
        remaining = max(0, _V71_CARROT_TARGET - int(state.get("plants", 0) or 0))
        changed = 0
        for order in orders:
            if changed >= remaining:
                break
            if available_wheat - _V71_EXPECTED_YIELD < feed_reserve:
                break
            if (
                isinstance(order, list)
                and len(order) >= 2
                and order[:2] == ["PLANT", "WHEAT"]
            ):
                order[1] = "CARROT"
                changed += 1
                available_wheat -= _V71_EXPECTED_YIELD
        if changed:
            state["plants"] = int(state.get("plants", 0) or 0) + changed
            state["seed_debt"] = int(state.get("seed_debt", 0) or 0) + changed
            state["lost_wheat"] = int(state.get("lost_wheat", 0) or 0) + (
                changed * _V71_EXPECTED_YIELD
            )
            action["farmer"] = orders[0]
            action["hands"] = orders[1:]

    # The second route-aligned cohort is closer to game end: its inherited
    # harvest already occurs within carrot lifespan, so only PLANT relabeling
    # and paired seed replenishment are required.
    post_changed = 0
    for actor, order in enumerate(orders):
        if (step, actor) not in _V71_POST_SLOTS:
            continue
        if available_wheat - 2 < feed_reserve:
            break
        if not (
            isinstance(order, list)
            and len(order) >= 2
            and order[:2] == ["PLANT", "WHEAT"]
        ):
            continue
        order[1] = "CARROT"
        post_changed += 1
        available_wheat -= 2
    if post_changed:
        state["post_plants"] = int(state.get("post_plants", 0) or 0) + post_changed
        state["seed_debt"] = int(state.get("seed_debt", 0) or 0) + post_changed
        state["lost_wheat"] = int(state.get("lost_wheat", 0) or 0) + (
            post_changed * 2
        )

    # A rewritten early harvest changes the shared order objects even when no
    # plant occurred this turn, so always write the aligned actor arrays back.
    action["farmer"] = orders[0]
    action["hands"] = orders[1:]

    action = _v71_pair_seed_debt(action, state)

    # Sell completed early cohorts on the next day boundary so they do not
    # occupy the shed until the terminal route.  Expand an inherited carrot
    # sale in place; otherwise use one free market slot.  The market cap is
    # never exceeded, and the terminal controller still liquidates at 708.
    shed_carrots = max(
        0,
        int(_get(_get(private, "shed", {}) or {}, "CARROT", 0) or 0),
    )
    market = list(action.get("market") or [])
    carrot_sale = next(
        (
            order
            for order in market
            if isinstance(order, list)
            and len(order) >= 3
            and order[:2] == ["SELL", "CARROT"]
        ),
        None,
    )
    if shed_carrots and carrot_sale is not None:
        carrot_sale[2] = max(int(carrot_sale[2] or 0), shed_carrots)
    elif shed_carrots and step % 24 == 0 and len(market) < 10:
        market.append(["SELL", "CARROT", shed_carrots])
    action["market"] = market[:10]
    return action


# V58 plants this final seven-plot strawberry relay immediately after the
# fourth public shop. Each plot already has inherited WATER, FERTILIZE, and
# HARVEST visits through the tomato horizon, so no actor or movement is added.
_V73_TOMATO_TARGET = 7
_V73_TOMATO_UNITS = 35
_V73_LOST_STRAWBERRY_UNITS = 42
_V73_ROUTE_RESERVE = 500
_V73_ROUTE_SLOTS = {
    (297, 7), (300, 7), (303, 7), (304, 8),
    (305, 6), (306, 7), (309, 7),
}
_V73_STATE = {
    0: {"last_step": -1, "active": False, "seed_debt": 0, "scheduled": 0, "tiles": {}, "decision": {}},
    1: {"last_step": -1, "active": False, "seed_debt": 0, "scheduled": 0, "tiles": {}, "decision": {}},
}


def _v89_tomato_route_ready(obs, action):
    """Confirm that the live-selected route owns all seven relay slots."""
    actions = _kawa_actions(obs)
    current_actors = 1 + len(_get(_farm(obs, _seat(obs)), "hands", []) or [])
    planned_hires = sum(
        1
        for order in (action.get("market") or [])
        if isinstance(order, list) and order[:1] == ["HIRE"]
    )
    planned_hires += sum(
        1
        for future_step in range(289, min(297, len(actions)))
        for order in ((actions[future_step] or {}).get("market") or [])
        if isinstance(order, list) and order[:1] == ["HIRE"]
    )
    available_actors = current_actors + planned_hires
    matched = 0
    for route_step, actor in sorted(_V73_ROUTE_SLOTS):
        if route_step >= len(actions) or actor >= available_actors:
            continue
        raw = actions[route_step] or {}
        orders = [raw.get("farmer", ["PASS"]), *list(raw.get("hands") or [])]
        if (
            actor < len(orders)
            and isinstance(orders[actor], list)
            and orders[actor][:2] == ["PLANT", "STRAWBERRY"]
        ):
            matched += 1
    return {
        "route_slots_ready": matched == _V73_TOMATO_TARGET,
        "route_slots_matched": matched,
        "route_actor_capacity": available_actors,
    }


def _v73_visible_ongoing_supply(farm, crop, day):
    schedule = {
        "TOMATO": (8, 9, 10, 11),
        "STRAWBERRY": (10, 12, 14, 16),
    }.get(crop, ())
    supply = 0
    for row in list(_get(farm, "tiles", []) or []):
        for tile in list(row or []):
            if not (
                isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and tile.get("crop") == crop
            ):
                continue
            supply += max(0, int(tile.get("yield_units", 0) or 0))
            age = day - int(tile.get("planted_day", day) or day)
            supply += 2 * sum(production_age > age for production_age in schedule)
    return supply


def _v73_integrated_value(item, inventory, quantity):
    return sum(
        _market_price(item, inventory + offset)
        for offset in range(max(0, int(quantity)))
    )


def _v73_tomato_decision(obs, step):
    seat = _seat(obs)
    farms = list(_get(obs, "farms", []) or [])
    opponent = farms[1 - seat] if len(farms) >= 2 else {}
    day = int(_get(obs, "day", step // 24) or 0)
    shops = list(_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or [])[:4]
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    tomato_current = int(_get(inventory, "TOMATO", 10000) or 10000)
    strawberry_current = int(_get(inventory, "STRAWBERRY", 10000) or 10000)
    tomato_drain = sum(_town_drain(future, shops, "TOMATO") for future in range(step, 708))
    strawberry_drain = sum(
        _town_drain(future, shops, "STRAWBERRY")
        for future in range(step, 708)
    )
    opponent_tomato = _v73_visible_ongoing_supply(opponent, "TOMATO", day)
    opponent_strawberry = _v73_visible_ongoing_supply(opponent, "STRAWBERRY", day)
    tomato_projected = tomato_current - tomato_drain + opponent_tomato
    strawberry_projected = strawberry_current - strawberry_drain + opponent_strawberry
    tomato_value = _v73_integrated_value("TOMATO", tomato_projected, _V73_TOMATO_UNITS)
    strawberry_value = _v73_integrated_value(
        "STRAWBERRY", strawberry_projected, _V73_LOST_STRAWBERRY_UNITS
    )
    npv = (
        tomato_value
        - strawberry_value
        - 50 * _V73_TOMATO_TARGET
        - _V73_ROUTE_RESERVE
    )
    tomato_shops = sum(
        shop in ("FARMERS_MARKET", "PIZZA_SHOP") for shop in shops
    )
    active = bool(
        len(shops) >= 4
        and tomato_shops >= 2
        and tomato_projected <= 9750
        and npv > 0
    )
    return {
        "active": active,
        "shops": shops,
        "tomato_current": tomato_current,
        "tomato_drain": tomato_drain,
        "opponent_tomato_supply": opponent_tomato,
        "tomato_projected_inventory": tomato_projected,
        "strawberry_current": strawberry_current,
        "strawberry_drain": strawberry_drain,
        "opponent_strawberry_supply": opponent_strawberry,
        "strawberry_projected_inventory": strawberry_projected,
        "tomato_value": tomato_value,
        "strawberry_value": strawberry_value,
        "npv": npv,
    }


def _v73_route_tomatoes(obs, action, step):
    seat = _seat(obs)
    state = _V73_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {
            "last_step": step,
            "active": False,
            "seed_debt": 0,
            "scheduled": 0,
            "tiles": {},
            "decision": {},
        }
        _V73_STATE[seat] = state
    state["last_step"] = step
    if step == 288:
        decision = _v73_tomato_decision(obs, step)
        decision.update(_v89_tomato_route_ready(obs, action))
        state["active"] = bool(
            decision.get("active")
            and decision.get("route_slots_ready")
            and not _V38_TOMATO_STATE.get(seat, {}).get("active", False)
        )
        state["seed_debt"] = _V73_TOMATO_TARGET if state["active"] else 0
        state["decision"] = decision
    if not state.get("active"):
        return action

    action = _copy_action(action)
    market = list(action.get("market") or [])
    debt = max(0, int(state.get("seed_debt", 0) or 0))
    if debt and step < 297 and len(market) < 10:
        market.append(["BUY_SEED", "TOMATO", debt])
        state["seed_debt"] = 0
    action["market"] = market[:10]

    farm = _farm(obs, seat)
    positions = [
        _get(farm, "farmer", [4, 4]),
        *list(_get(farm, "hands", []) or []),
    ]
    orders = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    tracked = state.setdefault("tiles", {})
    for actor, (order, position) in enumerate(zip(orders, positions)):
        if (
            (step, actor) in _V73_ROUTE_SLOTS
            and state.get("seed_debt", 0) == 0
            and isinstance(order, list)
            and len(order) >= 2
            and order[:2] == ["PLANT", "STRAWBERRY"]
        ):
            order[1] = "TOMATO"
            key = (int(position[0]), int(position[1]))
            tracked[key] = {"plant_step": step}
            state["scheduled"] = int(state.get("scheduled", 0) or 0) + 1

    day = int(_get(obs, "day", step // 24) or 0)
    for order, position in zip(orders, positions):
        if not isinstance(order, list) or not order:
            continue
        key = (int(position[0]), int(position[1]))
        if key not in tracked or order[0] != "WATER":
            continue
        tile = _tile_at(farm, position)
        if not (
            isinstance(tile, dict)
            and tile.get("kind") == "PLANT"
            and tile.get("crop") == "TOMATO"
        ):
            continue
        planted_day = int(tile.get("planted_day", day) or day)
        if day >= planted_day + 11 and int(tile.get("yield_units", 0) or 0) > 0:
            order[:] = ["HARVEST"]
            tracked.pop(key, None)
    action["farmer"] = orders[0]
    action["hands"] = orders[1:]
    return action


def _v79_rival_strawberry_flush(obs, action, step):
    if step < 216:
        return action
    shops = list(_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or [])
    if shops[:3] != ["BAKERY", "FARMERS_MARKET", "PET_CAFE"]:
        return action
    actions = _kawa_actions(obs)
    raw = actions[min(max(0, int(step)), len(actions) - 1)]
    planned_now = sum(
        max(0, int(order[2]))
        for order in (raw.get("market") or [])
        if len(order) >= 3 and order[:2] == ["SELL", "STRAWBERRY"]
    )
    if planned_now <= 0:
        return action
    seat = _seat(obs)
    farms = list(_get(obs, "farms", []) or [])
    opponent = farms[1 - seat] if len(farms) >= 2 else {}
    rival_ready = sum(
        max(0, int(tile.get("yield_units", 0) or 0))
        for row in list(_get(opponent, "tiles", []) or [])
        for tile in list(row or [])
        if isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and tile.get("crop") == "STRAWBERRY"
    )
    if rival_ready < 8:
        return action
    action = _copy_action(action)
    market = [list(order) for order in (action.get("market") or [])]
    existing = sum(
        max(0, int(order[2]))
        for order in market
        if len(order) >= 3 and order[:2] == ["SELL", "STRAWBERRY"]
    )
    private = _get(obs, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    available = max(
        0,
        int(_get(shed, "STRAWBERRY", 0) or 0)
        - existing
        - _v17_md_pickup_reserve(action, "STRAWBERRY"),
    )
    quantity = min(20, rival_ready, available)
    if quantity <= 0:
        return action
    inventory = int(
        _get(
            _get(_get(obs, "market", {}) or {}, "inventory", {}) or {},
            "STRAWBERRY",
            10000,
        )
        or 10000
    )
    sell_now = sum(
        _market_price("STRAWBERRY", inventory + existing + offset)
        for offset in range(quantity)
    )
    sell_after_rival = sum(
        _market_price("STRAWBERRY", inventory + rival_ready + existing + offset)
        for offset in range(quantity)
    )
    # Require enough integrated displacement value to cover the opportunity
    # cost of moving this inventory ahead of the inherited sale route.  Tiny
    # batches carry an extra timing-uncertainty reserve because one rival sale
    # can erase their modeled edge before the next inherited sell window.
    uncertainty_reserve = 150 + 100 * max(0, 4 - quantity)
    if sell_now - sell_after_rival < uncertainty_reserve:
        return action
    order = next(
        (
            order
            for order in market
            if len(order) >= 3 and order[:2] == ["SELL", "STRAWBERRY"]
        ),
        None,
    )
    if order is not None:
        order[2] = max(0, int(order[2])) + quantity
    elif len(market) < 10:
        market.append(["SELL", "STRAWBERRY", quantity])
    else:
        return action
    action["market"] = market[:10]
    return action


_V35_EGG_SHOPS = {"BAKERY", "BRUNCH_SPOT"}
_V35_EGG_STATE = {
    0: {"last_step": -1, "active": False},
    1: {"last_step": -1, "active": False},
}


def _v35_opponent_has_goose(obs):
    seat = _seat(obs)
    farms = list(_get(obs, "farms", []) or [])
    opponent = farms[1 - seat] if len(farms) >= 2 else {}
    for row in list(_get(opponent, "tiles", []) or []):
        for tile in list(row or []):
            if isinstance(tile, dict) and (
                tile.get("kind") == "COOP" or tile.get("animal") == "GOOSE"
            ):
                return True
    return False


def _v35_egg_late_pair(obs, action, step):
    """Turn only the day-11 animal pair into geese in strong egg regimes.

    By day 11 the first three shops are public.  Two early egg shops are a
    deliberately narrow enrichment gate for the patched egg hinge.  The
    existing tape already builds, feeds, visits, and harvests two new animal
    structures on these steps, so the fork changes the complete cohort rather
    than overlaying unrelated goose actions on the route.
    """
    seat = _seat(obs)
    state = _V35_EGG_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "active": False}
        _V35_EGG_STATE[seat] = state
    state["last_step"] = step
    if step == 264:
        shops = list(_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or [])[:3]
        egg_shops = sum(shop in _V35_EGG_SHOPS for shop in shops)
        state["active"] = bool(
            egg_shops >= 2
            and "YARN_STORE" not in shops
            and shops != ["BAKERY", "BAKERY", "BAKERY"]
            and shops != ["ICE_CREAM_SHOP", "BAKERY", "BAKERY"]
            and not (
                shops == ["BRUNCH_SPOT", "BRUNCH_SPOT", "FARMERS_MARKET"]
                and _clone_distance(obs) == 0
            )
            and not _v35_opponent_has_goose(obs)
        )
    if not state.get("active"):
        return action

    action = _copy_action(action)
    if 264 <= step <= 275:
        orders = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
        for order in orders:
            if not isinstance(order, list) or not order:
                continue
            if step in (266, 269) and order[0] == "BUILD_PASTURE":
                order[0] = "BUILD_COOP"
            elif (
                267 <= step <= 275
                and order[0] in ("PICKUP", "PLACE")
                and len(order) >= 2
                and order[1] in ("COW", "SHEEP")
            ):
                order[1] = "GOOSE"
        action["farmer"] = orders[0]
        action["hands"] = orders[1:]
        for order in action.get("market", []) or []:
            if (
                step == 264
                and isinstance(order, list)
                and len(order) >= 3
                and order[0] == "BUY_ANIMAL"
                and order[1] in ("COW", "SHEEP")
            ):
                order[1] = "GOOSE"

    return action


_V65_MAX_TOMATOES = 8
_V65_HANDS = 2
_V65_DECISION_STEP = 360
_V65_TERMINAL_STEP = 708
# Preserve a meaningful tomato shortage after our own projected supply.
_V88_MIN_POST_SUPPLY_DEFICIT = 300
_V65_STATE = {
    0: {"last_step": -1, "decided": False, "targets": []},
    1: {"last_step": -1, "decided": False, "targets": []},
}


def _v65_distance(left, right):
    return abs(int(left[0]) - int(right[0])) + abs(int(left[1]) - int(right[1]))


def _v65_tick_count(start, stop, interval):
    if stop <= start:
        return 0
    return (stop - 1) // interval - (start - 1) // interval


def _v65_town_drain(step, shops, item):
    shop_ticks = _v65_tick_count(step, _V65_TERMINAL_STEP, 4)
    center_ticks = _v65_tick_count(step, _V65_TERMINAL_STEP, 24)
    demand = center_ticks
    for shop in shops:
        products = _SHOP_PRODUCTS.get(shop, ())
        if item in products:
            demand += shop_ticks * (2 if len(products) == 1 else 1)
    return demand


def _v65_visible_crop_supply(farm, crop, day):
    schedule = {
        "TOMATO": (8, 9, 10, 11),
        "STRAWBERRY": (10, 12, 14, 16),
    }.get(crop, ())
    supply = 0
    for row in list(_get(farm, "tiles", []) or []):
        for tile in list(row or []):
            if not (
                isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and tile.get("crop") == crop
            ):
                continue
            supply += max(0, int(tile.get("yield_units", 0) or 0))
            age = day - int(tile.get("planted_day", day) or day)
            supply += 2 * sum(production_age > age for production_age in schedule)
    return supply


def _v65_integrated_value(item, inventory, quantity):
    return sum(_market_price(item, inventory + offset) for offset in range(max(0, quantity)))


def _v65_live_strawberries(farm):
    result = []
    for y, row in enumerate(list(_get(farm, "tiles", []) or [])):
        for x, tile in enumerate(list(row or [])):
            if (
                isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and tile.get("crop") == "STRAWBERRY"
            ):
                result.append((x, y))
    return result


def _v65_compact_targets(farm, limit):
    candidates = set(_v65_live_strawberries(farm))
    if not candidates or limit <= 0:
        return []
    access = _shed_access(len(_get(farm, "tiles", []) or []) or 10)
    first = min(
        candidates,
        key=lambda point: (min(_v65_distance(point, shed) for shed in access), point[1], point[0]),
    )
    chosen = [first]
    candidates.remove(first)
    while candidates and len(chosen) < limit:
        touching = [
            point for point in candidates
            if min(_v65_distance(point, selected) for selected in chosen) <= 2
        ]
        pool = touching or list(candidates)
        point = min(
            pool,
            key=lambda value: (
                min(_v65_distance(value, selected) for selected in chosen),
                min(_v65_distance(value, shed) for shed in access),
                value[1],
                value[0],
            ),
        )
        chosen.append(point)
        candidates.remove(point)
    return chosen


def _v65_decision(obs, step):
    seat = _seat(obs)
    farms = list(_get(obs, "farms", []) or [])
    me = farms[seat] if seat < len(farms) else {}
    opponent = farms[1 - seat] if len(farms) >= 2 else {}
    day = int(_get(obs, "day", step // 24) or 0)
    shops = list(_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or [])[:5]
    market = _get(obs, "market", {}) or {}
    inventories = _get(market, "inventory", {}) or {}
    prices = _get(market, "prices", {}) or {}

    tomato_current = int(_get(inventories, "TOMATO", 10000) or 0)
    strawberry_current = int(_get(inventories, "STRAWBERRY", 10000) or 0)
    tomato_drain = _v65_town_drain(step, shops, "TOMATO")
    strawberry_drain = _v65_town_drain(step, shops, "STRAWBERRY")
    opponent_tomato = _v65_visible_crop_supply(opponent, "TOMATO", day)
    opponent_strawberry = _v65_visible_crop_supply(opponent, "STRAWBERRY", day)
    tomato_projected = tomato_current - tomato_drain + opponent_tomato
    strawberry_projected = strawberry_current - strawberry_drain + opponent_strawberry

    candidates = _v65_live_strawberries(me)
    max_quantity = min(_V65_MAX_TOMATOES, len(candidates))
    fertilizer_quote = max(1, int(_get(prices, "FERTILIZER", 100) or 100))
    best = (0.0, 0, 0, 0)
    for quantity in range(1, max_quantity + 1):
        units = 8 * quantity
        tomato_value = _v65_integrated_value("TOMATO", tomato_projected, units)
        strawberry_value = _v65_integrated_value("STRAWBERRY", strawberry_projected, units)
        # Four appended-hand windows: conversion, first-yield separation,
        # three-yield separation, and final-yield preparation.  This reserve is
        # deliberately larger than the observed two-hand hire bill.
        operating_reserve = 3200
        direct_cost = quantity * (50 + fertilizer_quote)
        npv = tomato_value - strawberry_value - direct_cost - operating_reserve
        candidate = (float(npv), quantity, tomato_value, strawberry_value)
        if candidate[0] > best[0]:
            best = candidate
    quantity = best[1] if best[0] > 0 else 0
    # The patched curve rewards genuine scarcity, but supplying through the
    # hinge destroys marginal value.  Reject cohorts that would leave less
    # than a generic 300-unit shortage after their own projected 8x output.
    post_supply_deficit = 10000 - tomato_projected - 8 * quantity
    if quantity > 0 and post_supply_deficit < _V88_MIN_POST_SUPPLY_DEFICIT:
        quantity = 0
    targets = _v65_compact_targets(me, quantity)
    return {
        "shops": shops,
        "tomato_current": tomato_current,
        "tomato_drain": tomato_drain,
        "opponent_tomato_supply": opponent_tomato,
        "tomato_projected_inventory": tomato_projected,
        "tomato_projected_deficit": 10000 - tomato_projected,
        "post_supply_deficit": post_supply_deficit,
        "strawberry_current": strawberry_current,
        "strawberry_drain": strawberry_drain,
        "opponent_strawberry_supply": opponent_strawberry,
        "strawberry_projected_inventory": strawberry_projected,
        "quantity": quantity,
        "npv": best[0],
        "tomato_value": best[2],
        "strawberry_value": best[3],
        "targets": targets,
    }


def _v65_route_cost(start, route, service_cost):
    if not route:
        return 0
    travel = _v65_distance(start, route[0])
    travel += sum(_v65_distance(left, right) for left, right in zip(route, route[1:]))
    return travel + service_cost * len(route)


def _v65_plan_routes(starts, targets, service_cost):
    routes = [[] for _ in starts]
    for target in sorted(
        targets,
        key=lambda point: min(_v65_distance(start, point) for start in starts),
        reverse=True,
    ):
        best = None
        for actor, start in enumerate(starts):
            for index in range(len(routes[actor]) + 1):
                proposal = routes[actor][:index] + [target] + routes[actor][index:]
                costs = [
                    _v65_route_cost(
                        starts[other],
                        proposal if other == actor else routes[other],
                        service_cost,
                    )
                    for other in range(len(starts))
                ]
                score = (max(costs), sum(costs), actor, index)
                if best is None or score < best[0]:
                    best = (score, actor, proposal)
        if best is not None:
            routes[best[1]] = best[2]
    return routes


def _v65_phase(day, planted_day):
    age = day - planted_day
    if age == 0:
        return "convert"
    if age == 7:
        return "prime"
    if age in (8, 10):
        return "cycle"
    return None



def _v68_fertilizer_horizon(tile):
    if not (
        isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and tile.get("crop") == "TOMATO"
    ):
        return -1
    return int(tile.get("planted_day", -1) or -1) + 11


def _v68_needs_fertilizer(tile, day):
    """True only when fertilizing now extends coverage toward final yield."""
    horizon = _v68_fertilizer_horizon(tile)
    covered = int(tile.get("fertilized_until_day", -1) or -1) if isinstance(tile, dict) else -1
    return horizon >= 0 and covered < horizon and day + 2 > covered


def _v68_fertilizer_count(farm, targets, day):
    return sum(_v68_needs_fertilizer(_tile_at(farm, tuple(target)), day) for target in targets)


def _v65_start_window(obs, action, state, day, phase, targets):
    farm = _farm(obs, _seat(obs))
    market = [list(order) for order in (action.get("market") or [])]
    private = _get(obs, "private", {}) or {}
    fertilizer_needed = _v68_fertilizer_count(farm, targets, day)
    if fertilizer_needed > 0 and len(market) < 10:
        shed = dict(_get(private, "shed", {}) or {})
        free = max(0, 100 - sum(max(0, int(value or 0)) for value in shed.values()))
        fertilizer = min(fertilizer_needed, free)
        if fertilizer > 0:
            market.append(["BUY_PRODUCT", "FERTILIZER", fertilizer])
    action["market"] = market[:10]
    state["window"] = {
        "day": day,
        "phase": phase,
        "base_hands": len(_get(farm, "hands", []) or []),
        "requested": _V65_HANDS,
        "maintainers": [],
        "pending_hires": [],
        "routes": None,
    }
    state["window_key"] = (day, phase)
    return action


def _v66_append_pending_hires(obs, action, window, step):
    """Reconcile submitted hires, then use the first later market capacity."""
    farm = _farm(obs, _seat(obs))
    hands = list(_get(farm, "hands", []) or [])
    maintainers = [
        int(index) for index in list(window.get("maintainers", []) or [])
        if 0 <= int(index) < len(hands)
    ]
    pending = []
    for entry in list(window.get("pending_hires", []) or []):
        index = int(entry.get("index", -1))
        submitted = int(entry.get("step", -1))
        if submitted < step:
            if 0 <= index < len(hands) and index not in maintainers:
                maintainers.append(index)
        else:
            pending.append({"index": index, "step": submitted})
    maintainers.sort()
    window["maintainers"] = maintainers
    window["pending_hires"] = pending

    missing = max(0, int(window.get("requested", 0)) - len(maintainers) - len(pending))
    if missing <= 0:
        return action
    market = [list(order) for order in (action.get("market") or [])]
    room = max(0, 10 - len(market))
    count = min(missing, room)
    if count <= 0:
        return action
    # Existing same-step hires materialize first.  Recording their offset lets
    # the controller retain only its own appended actors without disturbing
    # inherited actor indices or routes.
    existing_hires = sum(bool(order) and order[0] == "HIRE" for order in market)
    first_index = len(hands) + existing_hires
    for offset in range(count):
        market.append(["HIRE"])
        pending.append({"index": first_index + offset, "step": step})
    window["pending_hires"] = pending
    action["market"] = market[:10]
    return action

def _v65_late_tomatoes(obs, action, step):
    seat = _seat(obs)
    state = _V65_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "decided": False, "targets": []}
        _V65_STATE[seat] = state
    state["last_step"] = step
    action = _align_hands(_copy_action(action), obs)
    farm = _farm(obs, seat)
    private = _get(obs, "private", {}) or {}
    day = int(_get(obs, "day", step // 24) or 0)
    hour = int(_get(obs, "hour", step % 24) or 0)

    if step == _V65_DECISION_STEP and not state.get("decided"):
        decision = _v65_decision(obs, step)
        state["decided"] = True
        state["decision"] = decision
        state["targets"] = [tuple(point) for point in decision["targets"]]
        state["planted_day"] = day
        quantity = int(decision.get("quantity", 0) or 0)
        if quantity > 0:
            market = [list(order) for order in (action.get("market") or [])]
            if len(market) < 10:
                market.append(["BUY_SEED", "TOMATO", quantity])
                action["market"] = market[:10]

    targets = [tuple(point) for point in state.get("targets", [])]
    if not targets:
        return action
    planted_day = int(state.get("planted_day", day) or day)
    phase = _v65_phase(day, planted_day)

    # The inherited tapes finish their opening hires by hour three on the
    # relevant days.  Appending here preserves every base actor index.
    if phase and hour == 3 and state.get("window_key") != (day, phase):
        action = _v65_start_window(obs, action, state, day, phase, targets)

    window = state.get("window")
    if not window or int(window.get("day", -1)) != day or window.get("phase") != phase:
        window = None
        state["window"] = None
    if window is not None:
        action = _align_hands(action, obs)
        hands = list(_get(farm, "hands", []) or [])
        base_hands = int(window.get("base_hands", len(hands)))
        action = _v66_append_pending_hires(obs, action, window, step)
        maintainers = [
            int(index) for index in list(window.get("maintainers", []) or [])
            if 0 <= int(index) < len(hands)
        ]
        if (
            len(maintainers) >= int(window.get("requested", 0))
            and window.get("routes") is None
        ):
            starts = [tuple(hands[index]) for index in maintainers]
            service_cost = 3
            planned = _v65_plan_routes(starts, targets, service_cost)
            window["routes"] = {
                hand: [tuple(point) for point in route]
                for hand, route in zip(maintainers, planned)
            }
        routes = window.get("routes") or {}
        tiles = list(_get(farm, "tiles", []) or [])
        inventories = [dict(value or {}) for value in list(_get(private, "inventories", []) or [])]
        orders = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
        access = _shed_access(len(tiles) or 10)
        for hand in maintainers:
            actor = hand + 1
            if actor >= len(orders) or hand >= len(hands):
                continue
            route = routes.get(hand, [])
            position = tuple(hands[hand])
            inventory = inventories[actor] if actor < len(inventories) else {}
            fertilizer_needed = sum(
                _v68_needs_fertilizer(_tile_at(farm, tuple(target)), day)
                for target in route
            )
            if (
                fertilizer_needed > 0
                and position in access
                and int(inventory.get("FERTILIZER", 0) or 0) <= 0
            ):
                orders[actor] = ["PICKUP", "FERTILIZER", fertilizer_needed]
                continue
            while route:
                target = tuple(route[0])
                tile = _tile_at(farm, target)
                if position != target:
                    orders[actor] = _v20_move_toward(position, target, tiles)
                    break
                if phase == "convert":
                    if isinstance(tile, dict) and tile.get("crop") == "STRAWBERRY":
                        orders[actor] = ["DIG"]
                        break
                    if tile is None:
                        orders[actor] = ["PLANT", "TOMATO"]
                        break
                    if isinstance(tile, dict) and tile.get("crop") == "TOMATO":
                        if not bool(tile.get("watered_today", False)):
                            orders[actor] = ["WATER"]
                            break
                        route.pop(0)
                        continue
                    route.pop(0)
                    continue
                if not (
                    isinstance(tile, dict)
                    and tile.get("kind") == "PLANT"
                    and tile.get("crop") == "TOMATO"
                ):
                    route.pop(0)
                    continue
                if (
                    _v68_needs_fertilizer(tile, day)
                    and int(inventory.get("FERTILIZER", 0) or 0) > 0
                ):
                    orders[actor] = ["FERTILIZE"]
                    break
                if phase == "cycle" and int(tile.get("yield_units", 0) or 0) > 0:
                    orders[actor] = ["HARVEST"]
                    break
                if not bool(tile.get("watered_today", False)):
                    orders[actor] = ["WATER"]
                    break
                route.pop(0)
            else:
                orders[actor] = ["PASS"]
        action["farmer"] = orders[0]
        action["hands"] = orders[1:]

    if day - planted_day == 11:
        positions = [_get(farm, "farmer", [0, 0]), *list(_get(farm, "hands", []) or [])]
        orders = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
        target_set = set(targets)
        for actor, (position, order) in enumerate(zip(positions, orders)):
            tile = _tile_at(farm, position)
            if (
                tuple(position) in target_set
                and isinstance(order, list)
                and order
                and order[0] == "WATER"
                and isinstance(tile, dict)
                and tile.get("crop") == "TOMATO"
                and int(tile.get("yield_units", 0) or 0) > 0
            ):
                orders[actor] = ["HARVEST"]
        action["farmer"] = orders[0]
        action["hands"] = orders[1:]
    return action


def _v66_output_at(tile, operation):
    if operation == "COLLECT_FERTILIZER":
        if isinstance(tile, dict) and tile.get("fertilizer_available", False):
            return "FERTILIZER", 1
        return None, 0
    if operation != "HARVEST" or not isinstance(tile, dict):
        return None, 0
    quantity = max(0, int(tile.get("yield_units", 0) or 0))
    if tile.get("kind") == "PLANT":
        return tile.get("crop"), quantity
    animal = tile.get("animal")
    return {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}.get(animal), quantity


def _v66_capacity_reserve(obs, action, step):
    """Keep projected late-day holdings within the 100-unit drop capacity."""
    hour = int(_get(obs, "hour", step % 24) or 0)
    if hour < 18:
        return action
    action = _copy_action(action)
    private = _get(obs, "private", {}) or {}
    shed = {
        key: max(0, int(value or 0))
        for key, value in dict(_get(private, "shed", {}) or {}).items()
    }
    inventories = [dict(value or {}) for value in list(_get(private, "inventories", []) or [])]
    carried = sum(
        max(0, int(value or 0))
        for inventory in inventories
        for value in inventory.values()
    )
    farm = _farm(obs, _seat(obs))
    positions = [_get(farm, "farmer", [4, 4]), *list(_get(farm, "hands", []) or [])]
    orders = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    outputs = []
    consumed = 0
    for actor, order in enumerate(orders):
        if actor >= len(positions) or not isinstance(order, list) or not order:
            continue
        operation = order[0]
        tile = _tile_at(farm, positions[actor])
        item, quantity = _v66_output_at(tile, operation)
        if item and quantity > 0:
            outputs.append((actor, item, quantity))
        elif operation in ("FEED", "FERTILIZE"):
            consumed += 1
        elif operation == "PLACE" and len(order) >= 2 and order[1] in ("GOOSE", "COW", "SHEEP"):
            consumed += 1

    market = [list(order) for order in (action.get("market") or [])]
    planned_sells = {}
    planned_buys = 0
    for order in market:
        if len(order) < 3:
            continue
        quantity = max(0, int(order[2] or 0))
        if order[0] == "SELL":
            planned_sells[order[1]] = planned_sells.get(order[1], 0) + quantity
        elif order[0] in ("BUY_PRODUCT", "BUY_ANIMAL"):
            planned_buys += quantity
    actual_sells = sum(
        min(shed.get(item, 0), quantity)
        for item, quantity in planned_sells.items()
    )
    needed = max(
        0,
        sum(shed.values()) + carried + sum(value for _, _, value in outputs)
        - consumed + planned_buys - actual_sells - 100,
    )
    if needed <= 0:
        return action

    prices = dict(_get(_get(obs, "market", {}) or {}, "prices", {}) or {})
    sale_priority = sorted(
        _SELLABLE,
        key=lambda item: (item == "TOMATO", int(prices.get(item, 0) or 0), item),
    )
    for item in sale_priority:
        already = planned_sells.get(item, 0)
        available = max(0, shed.get(item, 0) - already)
        quantity = min(needed, available)
        if quantity <= 0:
            continue
        existing = next(
            (
                order for order in market
                if len(order) >= 3 and order[0] == "SELL" and order[1] == item
            ),
            None,
        )
        if existing is not None:
            existing[2] = int(existing[2] or 0) + quantity
        elif len(market) < 10:
            market.append(["SELL", item, quantity])
        else:
            continue
        planned_sells[item] = already + quantity
        needed -= quantity
        if needed <= 0:
            break

    # A product/animal buy that itself crosses capacity is cheaper to defer
    # than already-realized farm output.  Seeds do not consume shed capacity.
    if needed > 0:
        buy_priority = sorted(
            (
                (index, order)
                for index, order in enumerate(market)
                if len(order) >= 3 and order[0] in ("BUY_PRODUCT", "BUY_ANIMAL")
            ),
            key=lambda value: (
                value[1][1] == "TOMATO",
                int(prices.get(value[1][1], 0) or 0),
                value[0],
            ),
        )
        for _index, order in buy_priority:
            quantity = min(needed, max(0, int(order[2] or 0)))
            if quantity <= 0:
                continue
            order[2] = int(order[2] or 0) - quantity
            needed -= quantity
            if needed <= 0:
                break
        market = [
            order for order in market
            if not (
                len(order) >= 3
                and order[0] in ("BUY_PRODUCT", "BUY_ANIMAL")
                and int(order[2] or 0) <= 0
            )
        ]

    # Once saleable shed stock and avoidable buys are exhausted, allowing a
    # low-value harvest to execute merely replaces it with an arbitrary engine
    # discard at midnight.  Defer that production instead, keeping tomatoes as
    # the last resort.
    if needed > 0:
        output_priority = sorted(
            outputs,
            key=lambda value: (
                value[1] == "TOMATO",
                int(prices.get(value[1], 0) or 0),
                value[0],
            ),
        )
        for actor, _item, quantity in output_priority:
            if actor >= len(orders):
                continue
            orders[actor] = ["PASS"]
            needed -= quantity
            if needed <= 0:
                break
    action["farmer"] = orders[0] if orders else ["PASS"]
    action["hands"] = orders[1:]
    action["market"] = market[:10]
    return action


def _v88_late_plan_active(obs):
    state = _V65_STATE[_seat(obs)]
    decision = state.get("decision") or {}
    return int(decision.get("quantity", 0) or 0) > 0


def _v88_late_tomato_overlay(obs, base_action, step):
    """Return byte-equivalent V84 action unless the public decision is active."""
    proposed = _v65_late_tomatoes(obs, base_action, step)
    return proposed if _v88_late_plan_active(obs) else base_action


def _v88_capacity_overlay(obs, base_action, step):
    if not _v88_late_plan_active(obs):
        return base_action
    return _v66_capacity_reserve(obs, base_action, step)


# V107 changes only the timing of the day-0 WHEAT5 product order.
# It is an empirical policy candidate, not an exact V92 rejoin.
def _v107_index0_opening(action, step):
    if step not in (0, 1):
        return action
    action = _copy_action(action)
    market = [list(order) for order in (action.get("market") or [])]
    if step == 0:
        hires = [["HIRE"] for order in market if list(order)[:1] == ["HIRE"]]
        sheep = next(
            (list(order) for order in market if list(order)[:2] == ["BUY_ANIMAL", "SHEEP"]),
            ["BUY_ANIMAL", "SHEEP", 2],
        )
        sheep[2] = 2
        market = [
            ["BUY_PRODUCT", "WHEAT", 5],
            *hires[:5],
            ["BUY_ANIMAL", "COW", 1],
            sheep,
            ["PASS"],
            ["PASS"],
        ]
    else:
        market.extend([
            ["BUY_ANIMAL", "COW", 1],
            ["BUY_SEED", "WHEAT", 7],
            ["BUY_SEED", "MELON", 12],
        ])
    action["market"] = market[:10]
    return action


def _v125_v113_agent(obs):
    try:
        actions = _kawa_actions(obs)
        step = min(max(0, int(_get(obs, "step", 0) or 0)), len(actions) - 1)
        _observe_opponent_market(obs, step)
        if step >= 708:
            return _v20_terminal_action(obs)
        action = _weed_repair_action(obs, _copy_action(actions[step]), step)
        action = _v17_feed_guard(obs, action, step)
        action = _v17_room_evac(obs, action, step)
        action = _repay_shift(obs, action, step)
        action = _rank_sell_slots(obs, action, None)
        action = _preempt_shift(obs, action, step)
        action = _v17_r5_counter(obs, action, step)
        action = _v17_md_counter(obs, action, step)
        action = _v38_farm_pair_tomatoes(obs, action, step)
        action = _v73_route_tomatoes(obs, action, step)
        action = _v71_route_carrots(obs, action, step)
        action = _v79_rival_strawberry_flush(obs, action, step)
        action = _v88_late_tomato_overlay(obs, action, step)
        action = _v35_egg_late_pair(obs, action, step)
        action = _v17_room_guard(obs, action, step)
        action = _v88_capacity_overlay(obs, action, step)
        action = _terminal_liquidation(obs, action, step)
        action = _v107_index0_opening(action, step)
        action = _align_hands(action, obs)
        _record_own_sells(obs, action, step)
        return action
    except Exception:
        farm = _farm(obs, _seat(obs))
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_get(farm, "hands", []) or [])],
            "market": [],
        }


_V125_ACTIONS = json.loads(zlib.decompress(base64.b85decode('c-rk<%Whm*a{L#rxlld$@{TRlObf$q3Y0X3af4_y;29V&#*4OhhX36%A62*RiHwZMbBdy7v?~;g_n!C3jEs!@<$q58_S^4&|LgB3|MJVp51-%Py!-XT^~X=2?=~m*rzijZ+kgJse|`DOmydt{?f3uq>wkaw{L9IY?;rlEefZ(?Uw*y*`TbAVHz%hjZ*R9Jr_1K+k3ViUA0~hJxY@k>^7Zz|&Gn~~(~H^HKW%Pqe?B=~?0){??)Kg1x1aX^aejaQzo*lVeR%)&PoF>R-?W(Y?U$47=HsV_w*Gv3_vy!nPrI*X9}WlN<L2gO|JK#~t<R5}yb3gA`r7@c`Bb0=%w8AH9_-<+B@c75IOyxsugJST++4riMB|D2^ZXCsZL@Zhw?6&LbUd4OJbd@_elZ;M^=YPppQR(bxt_m&zdWu#ZSLlaX#U;d>VZplIbTE{Z$Hl$QM)+*@c%pG;G0?R*i^QIb2z}WQQG(K?e){reE!kn&YX1Jn#=ugwJ&`hh3T)-=>q!?O%B)z%?c)Oc^Z2#W|QG)W~}{<K4VYgPKWN`x$~X3AHsH;f_1qN4mYqF!lRX+EeBoDMiw19`Q$yfR3A(En|vO@5I&tSV2-kR(+6?)j@^eZXYWV!!5g^!xc5Bx>6diU$3CA<_>c}f|L^2YL!X;|_zI7m-72fVnoJH;;{qA;)cM)!Y~L4e!Q37pKW)s2F)es|dvmjS`{~y|Z0<h2zj^<!hiAf|!7IPSSR&<j9BB@oZ|zBY!acNeL}ot@uJW^c!vcKO>))8)c^}tx?>4pnI&BhQ-Zkdq#0Uoqx8i32V+8IA+^eUhZJEiu57XXeeM|=sIQE7?%3KxrDSIFr3-l>{AoB=B`?13xjhkF_pyEN5Y+q#qQQthDf8y!%xxNbUls*o6%ZBp+jQjl~TVpWa{4H=oY|FfT*5guBmEdMCY*>GP+W4o*_dc+p))>HEbQuK*NES^EcCqzcaSY7~Zs*iGC|rY?A<zlcNf*Nq0|DWT;iXZ$8yUF1Kd$>m1-#68G&L5$TcY_-Zv@STD48c@c(~P;Kbiv608X<2K!S785gBqo!&N%;<ey_{|2W9m_s4#Jtcgj*){7mh2f_3pDxX+dXO%Ov;)|Q$NYUjpz>vP^VP<z#3=|Jia@tQq;=NudyVHZQ`SJSh&r!#E0i#KGbXQ*tL8GDC^`$sO)3NB`2egCJ1`u~lfG!k-5BjcSk8d@d8DNj>24y-@IhFyyk%MKoU!%{1vJY3vgT8+vx>Tm`o7-1bboiNIL2oYbhDvz2eQ?Wh`feb6e{3(pbLQAcXAgV+fW~*JgFc}V^<*l2{B(D{{b6%=_ZPqpl;TG05(^s+-oCi+fkN^a(_u@71edn@k?b2$di*RMH^VTT!&m)~k`cvVL8tApjHao*F@>l+%!miGy4J_;!|syKAE&{v&v)!(8)7c%z{q2lzcC+y;wp&xZGHXN%&LutK0P+H67g*BEy6z&sMF5lDmdTK*m2+EOkXQ%b#B`<FJ!62=s~;Fo?i83Bi`MjbO{q^SNvn@erNbf<z8U`#N-y--rn9kq^Uq7>gkV9GxX(vd?!Q#_rAClt}D}rPT{0B+%iT^7{nHt4{CG)$VTk*<j6x_gU*1V1G0Wb-}0klU?lOSa@k6#P*(s|cRc!-Ms3_T1+FA8wfR(rA2)@tB4{E&G54P;U>ySeCj28{F?MeSV+7hcM<?3+Y^dx7HrD8qIWpk<X_4J|?77CH0<Oi(N<Ee_bY96pwS#65ae-m8b#7e298ufiIu4~MvzhkGrJ?NXdgC1iD8ti@jMaFR?aY7>P-;#FUuWhThY8dZ1bF9p+V5#OBU(<^CAo+hJ!O{j>RE~?516iZp4Qu93hDUof==+Yt<9jKi`JFB57${FDbomT-`k!J;*j@6)GezTw~k%sx=k3~szF1H%NOjmcBhl;9kginw_+kIyZX})B|i}KH|$ig!(@&t-SllYGTqa&4O!%rN*92%nTf_s$=z#GT+(CpwL`EwTj!Yxr-MR`pqqOQ1Uaf*dpB6`@?(ox8^C%v_A7g0#Kl(vs)goN^i%5=Z?KclK%>t`+fVOr{&eU7LH7t<Iq+vJs=RyG7=P&MUf;jy;@|yI$8SXYA@GbK1!uS2bGyHxL@CTi4j0vd10D6QxPSW7)0>s8bg({x_hpNPc&Dd48&BLZ!6uN15v_3!BzKTW-kR50_lfk9>|#=qvbwOqSHM6kVb>ayqP?~EoJne<V=i79lp`J^3)GWX6PLyk;rr6E-G6HPz#VH$sI1I2b2(9y$rvM;*PLC#y9B|cz#3N&d`jo2DkE`1H`~IXEns_@gW2A4yJv#Iq{m4x(k;K)@W=`5V)~^h3ex_*3y_&^IK^EBc3HPiOPa^dEz_xnv#ERjAoU!-gf)=)z2h(f7CvGxZd=FNboiT>aP1gIA^*oPMpY{|Dc~#nU$>90P`=DoDELd%=aAQ<)90j?^nA$LxVRr4Vmu^3uSwKVccu+F+39<c4j=iQrX*Evb&FfI#-Q%y6W*>D56x-azNZhGqNUkUZ3yUwtt7MpdnrX>hl)s3f4(PxTw99+ggUf1X9eQ$luRegv^8E|z^3btE>jO%Pl}yU;@0=NyVZ0(q06yAy{QJ3wVIqrOj1MOp{bGcN)Qp8N9}r{gSU}ljyhjyeeY1Y;#g|+nDpE+b`KHzOw#qvR!2`@QF0pTSCV6HfX%0pNkzOl=ABDEQtmBvt1f#@(_N-wf);3DNTm}SW`tUqydcJ{W@o2m!-wucO%aZ37>y4vs=&Dt(1rZhz7`j>BG_YLWrbshY_}Uj|IgYTJrVlX*(zW_lgG6biq|k$Ey-dmIw0f#$^b#xQO+qn-I4u)ZR?8F*9FvMkYTrErk-yhRL^{}iqU*=-zQeAbVnZaqmWDWD825ya$UfIR8bmyCvFxDq`&}vuiU`ymD!tUNSy@PD_Wr>MoryL18iv#tlGrV0h+!f3+xY)rdA)#ngXMDO69UC%zNcBJD`42xS;T<ioh?j5(E6b5P5|gmb>_%h2Wpl<}OOEGWGEBQ9x1#6X5KixtDEt<Lq9)1>&T@XyvdV_x^7%{afzAdCbXN7{&!_GyRH=ioB9w$BcBvSY0h)B<xsv(^8B?&JNS%rvF^+9%Y`xR&J!pQ0{t!nRWCnu4lP8%hx1BRc8Zn9!ziCWtwxGcX0aN<RsCuYlRk`YI;Oy1=nVDQF5Bd@`6pQQ-_;=<dUb9iHc1(KKjJ2A=-$lE9Gg!ws>VQ&{{I2uyo#t&^cksgj$S4F_FV|FT2?iGh%d*iv1{bGAP|(xhl{n&O<n*C5+2y6qB>>7$l0kWU%FFADvFIM;N6DeNomXk1-Y*<G_y;AQ|br&oaTA>JG(dXK!7PA`v9LFxs$@UKv_haB3nB3vH#HDAbT{S;o9s)~@xG_k>&b^1cEGr2rnV5eQk5N98MIfOfesUqA~OhvtsWPh?gxAQEXb29?lITt5Z2K9z?kE)X^rp-xpY9c1}XUs7tY0Ex*9X=#X<frT&8?h`%x4d>!(r%*B<j3n3;sSkPjq5=h1?XNB=;(aj;5X{g<$eEy1O!N=cJAn`bS*Rc{mBkN&0c%Tf8-T=r&;pJzfn5?8ndsH^rzI8IRiL(<YqgcCDU@H6T-sbhuDD*5`lyvNP+ur~6&Z^Mw47)BQFReMJMqt|4HV&aQyOH5=`&o^LL$!zr$q;j(=UoK5}w(8Ic?8ng60h3#t>H<fHce-&9mWa9xzH}ap*BJ#A1>glAM(q`-Q|c+Gt}<UdT^U8j@*4fufqT3s#1%vw}XA3$78zWJ^jb=`}=&DKA*KSN9d(6NP2SLY@CekbE};8gzmdM>i@04NOy$hMP3K6F`KhX>CZU??o*+ID^YB5=%ON8EItGkw=5%NGN*_HUXqCH{a--kM+Gi*WAlMNdwGdRRiRgI2c$9%fhH@<{nGiEbYgI;rmnXR)YZ`L(gv{78>SW7=wXh>O{P*V>c7g3mRYoc+UBhYdMv~b(U#kD>2y33Pg<X^;u{}9BJm#*XcwdSUR|cozRr@Y4o5nAi01{Rfm-%snA-b09#;)CZS;P_Sp{nV@C94L}9v^a<&>@e_Cen6}kCmC0fRmb9YqNf7{gH#f;<hE)mjT6bxYRlkWah4EW35ZzZ8m%DTu@_!7o(A&mgcZ!2;)Mpq+qXgL+36Dev}i3U@NmZZ*!pyK!mJ0cswK{+q!{sC(R0_(XbehuXTmUa8L40JL8mO_d#rHx8HI1MKjRv^<76jdw}|Ewd4iTO%e22MVS1^9$b6mgnp8p5DW$`2H!zHG57lH|XuaKlDPo9c+$>vft{+uKwH6~>e5l2Q+X`Me;p$g5)}1}C+O3@k<;`f+(k10V)1UZDg?)xdm7%dj&Ns>(wa1S{E)5J3kxn?p+dwzFn^$lLr%qT1tY#7&BY9yq*Lk#F2O!mF?62=jCV6ml5o(E<%;P~yJ9y2q%Wfy)!dvZ2Fg22QubgVO^7rOFI-D?n_EBj%A1kixz;fSpyxG93_oeDnyi#ha(`;x=*Z;GlZ*IJ8gSbj2VQn!N5elA~c}_7`18NyjC21dgO~LPE*eEcK~ch}jZ_A*UMk)NyhK-4q>SiT4gRk*Ho!?=5jQx?I^{d9g5lX8}~^FBm0Oz3~_8K_6}%LjbkkwcI%Q@`YqWjxHU)r&Ie*9ON*cz;l@G7@tOxujY_&-!=u6YbD$NX-T=Vk@8w1c1UO=@qPy7P#+<fCka^R*0)OOWtpqlXbTr%*?Uk}znF|PB<Pofh8tM5ll9xD-9Mzq1!mEkP$%coNOu$yzEg`M_Q5oQtRN5c097pv2C|f62iBKUB|=g@aXEMt=CR!?U{1v0WPz}0aGB|=rwB;v09GQ`#xz_R%vf;b(Re&Q8F6R@{MMfWLXrj<*^D_3wmzCV$<U3+RXUS8uOU!#JGqq{Qo>3K1;AP7VWVUjL6W~u@;f&O)_A?r7Ar?h<@7;<ace^WOCp-(yH-e9E_G$;(1I1Y)81O6ee#kQmUn4M@sVscNpyll!_Ew`!V9SYmB}}U+nv?S6LLm+bWYX7kEGqxwB#H(GB`$}|7`SAN;z=yck0>eaWPE!{_U7wEk^+{yM;laq>32LKk7t-<Lce}KiawH1M0l=c5Re+;o?#`9H$C|=$3o6l!02gh%w{=9KqQ5^^J?Pfy}`@60jt52`WVjhfv#x06co@kH9q`;6OI(T#Z{x5@+JnvPeJ_gb~Xr)Ybsy?b5ta2cuL}8x4U^U8Oh=KGQHHtHmg0|3m+xT7bd8E%|We;VC68ooDS+TKZ#`dQ(Lh{XDDp@YtPWu*`Y3d!<^Wk|99cCu(T3yQGK+3DOZldK?GSOV&J7;NUo#Qf;M_>n2rz36PlrvxYL_Ic!JVyuw&$ns^=hbHybhx$5<V<Ss|MKS{bs)8JeOuQB!<4IoNP-L*BsSHGIK1_;$vhWD`q%w$r;?2IG;<YJk(wy1MPjRW0vaR?rjb3YAu@Gp5Jw_s1mPGR<U!HGT=uVTDtp6o;x3NYyI=A}_bjK<q?4?z@3&PG4tb2y6ft?32+Cd%~Jiy4Su`?<lW52k~iV7lxV3>K+ulnY0W-d;2`<5kIZP`hzyFlhLRS|}cvq(LkP7A)nw07@T-P6D5$wKCj6PPB;FG873@lX!z%KVzX#r;siLP7SJ5I_wh>%<@yffX{wVk4ml_C*5A@o17wsPG5haNZNOKS%(yMDCVeBE&F6x4T&|}M`vJ{6^EP#(Ip~6D7oOT1T@r}j{*Sabtf$ZQCPzSdjJXWTA^n;*b*6P2Lqh;A~+-j&&p}EVfr?G9@#Bf>N*jb0}*m`rL?baTZS$h6m36Z1pOjFUKYz&mZOpd(McySUM@QYy7=6HqV+*XW}^(i0DN_E%Xx0S&Lnlkr~y}~wg|n8<7TmRdcbLS#cmu&&t67gn0@BfmaOGaC)EfU)^50;8(+Q%CB$L((61?nV_01(cgBfw*pD1ug?9HKy*n}F=u`>84dpVZ`U#TAhLZ?pwp?W&ENZnRke~*1U3XrW0%`9aEl<tXKTS*o*U4E5G;k9t*-DXq<SN*R=C0zroIME6M3@@Fb3XE^5fmm_k1j&Vs&sp7l>`F5{^`+ULvKq$?9KUh?KT3-s`c@rEWC{%vMJYqjB@i20-$SQhY`<~lOY&!fYwX{M<LUFpL^=<`B$$E0OHU|QW=2*6{><aiDp_si?p<5O9zx+qy_M5XXTZ1Rx-qd!YWT8pD9>ym8B4{ZZ!`o9;BeNG>e1fK4^6aY44&<mM=F6)G{(R^fu2IS}(?Q04S|-9fPj}G3^I*T;jzTDo<jKFh?Wc!Hh<fGLDff?dK~Sj46AZ*y6C7QEtM^xeLcg2&j{D>i;<+Li0wjtvV7C45Uzys`pkW45;=P(kyX>)n1GMA~9n-0dm7KQ^~AK%bLe$Ma+ZjUx8&a*7ecVOx=`~%)pi>ujcN66v9xlPE^=;l`4%BCUk-p#xkX1CW(-bA>dz<Ke*^SR~g_ItT^fTwC{(r#v8wM=F?$gf^}OgIGQZ{G3O_X4N(4BR&`VquyM+MIy-P#VP|UcBAQx{e_~#iHrv(`h&^0MD(X<V@)ioB%$PKZG>uqgG-E`I*yUB%aUp5<ME*jNG}J3h*EwT~+>KJj9sYGpS7cIvB>ur>$I?$3BYslWoX*lV>0RvzYK>F^Sqpd=r_ZFBj;Bdr7gPfsxK<gFtYNWH2oK<{=lC&7sTNN*r2|hd64O?1^rkGAuTN)rTrrrz3NIMZ(ksNI7SaI9S_ZeIu$=)16>_{<s=!vej^t7=RW+AB0B=~))wG?+YSoEdFPCPR+2OOSu&T^==!JOPur{dCfR7~++l`uS)m=IToL3CTi*O~L*fXZ)6-^W1ZU8=lnUr}^t$_*`1_?z0@8aNe4GY|K62m5EN&pyY103{$As@9xz`>Geq2_RCl=9eFVhbZaNwFGq0Jr^D29iNm!IUw2OazHA=9aZ$*Y|yyo+|qz4>+bcP7sf%%0z>aqp1t&R4`>4Yfd6M$ok$%tBZD^mKd097%fOKtydX<QQGW*jxzW@ojp&yIZqCYW^Cyg(Hx7WF(VBUHj>-~e}FDN7GHSw@+Ztgd3Fc}SDBwxRZLn+r6v7fGHiKw&I}#st#|E3r(8dJHa`q}jv~_gB?+QM))>K&(p_?SEWl`r4lU1L6U*$Cfk3>57Nr@ybPX-&Zn)W5$0(Miq)W;YQ9?l$+5TtALBK)k7KRf3b40mkrP^)co2Fcg{yJLJaDBMSSEU#+<9B8Ge}ODy6GT`h)`V(1G|$WkaQpHWRqjRtZ|lfpF<VyN18L;ttQZP}MRDSTv$eV+n<goYRddl~;+#~)OLoHy*tNAilT_F`g@P1kQkmbBD2ZdgKAZE(qd;m#9??M3m}LB%D<eO4e3#S<OKJgSioq`Jv$!fxJMv+PG8;`)3u2Jr<y7YnUJ4hK!LWnpZ5Wvca~ZmjhSZS9{#3*G0vDD*%wXQ}`4fx;-5-Oh><Ajl5)kO{nyFHRI-TpzeVbG)@F!7~Bacq<F%#G>$K<F{i<RrHT8YLyV&hS5p=8ns7`Lk@#mPp4iA2qny5CiSR9Mrv%RDHkYBW8YfM3q~yW0`#H>p2$QIXY4>{O+0I13V}nw-m=7xpxaH93DSddLGZ0z<#==sgbcZMmi?4+kl=z%)NQHrjcH@XRxccAgE$snxGLVv)}qwwgJA=+6h1Ue1z(&)^(07cFJXs|?tKhhS^QmylCZvlpn*0S%xGaZE&|G~IU#YoroN4iv1_Lpm0*jf4W&bx^D-5XV_-C9D9)0?8M(dDcYgg0)Nqsc7Im2IF9TE1fWjjk|7Coy#R9sqaknKUN4BwS;qJ#}xi;jwhr3w1FBRwoo-;jj;(W8Xjt}obXQynbB@YwZ<%N7QCA;WI~)=dR#9ePmE|Y-cL(bN?J-8hN!Vw`Gn{u8YBXS>W{kyJ$t7rTyLOmqQldTD=|JJa6DCXLVLlOlwE(Vsxpt1Vo0?JpI3$o?XMlbRV-m0(~Nu}pE##2CGLV%)H&~h6n!%h+JdB<IJV0rLfC;LzZgH`jIStwNSN){hC`n6;ViL_w5Z!Uh|yj&^r=<7syi*NCS=}AMLoOlo%6Byl_XfpN(8WN6IoHUP8tOz+mYWj9Q1_;X2Ti?$CPyJV-#M|EpaGGcnax||9Ip>nX@B7D)R;|!Y{Mz2>4IDXmUULrm6XXL@%(hMpm+H-A+}@kFzgj^*alIfR7g-Hb-8`d~{=v))0$oGOwsQXTJesNx8Qg@w(1xva?2M4k~t?)%`0cKv1n@15iuoJxLf#&(jV0Gqnn!=N;4;g(@U-&#4VXCEwFouc`ngyvuaT*fI|4`)QqXIi=(rPV}5;AVXc;UGub97b%tL$&)7vI^d?a)pWLtq}JDdh!ggu61Afh7vGeM^l2o~8n}Kec2aU>ktyTK2JBSq14cS$tD;v*R%gn$wCO8-ET!19$y-PhtfeZNJc8+5=;wVGx=|F^<-4Nt(wE-)KJQGd&}0Qz+A)Y)gvF*6BfM8I?Qm`sz;?n4jG<h|j}TxEDzk?-m!=7)l#3GklWd~%{>TC4zojB64Xty#@hdDVI4>zqD{?ZuQ;;h49gmvFS)H$_R_bNFR%&0i^l@sXZiQ@E6yJCqHB)ufw-n>+t%j;s|5Qb*OxW)wYM|27oI_%r^-Tk-N8=^xB3%`(8c8jd$l{f;1}?Wn07L-0Ah#!FJcw#NhM}m4lB?B3+mo@#<qK?u%4{a3Y}2t*A}JgO6p~rpu~S&5TzW`nh&h8rV7Uw@OtMZTExp<_(i9CQ`y~rErsGEDGdM&XJOThXDY>d9;Fa-Iwfd2{yhunHv>2q&EHTn0iP~SgfDogGE|Ds0pj}IHo_1Cmm$6zTvgN{~E;*UJ(R4B3l!MGty;)1^KPfm|rkp{!f1GQ`7FwF{Cko<$gCUO>1)&SAgtl`GoQd!ToWW!_Kw&(E>03JzRu+jh+9BzTu%LyvNs@r|QTb3Stg!a<&GSjL{9fDh<!gJkS)<=jkxvUYc+nD`EZ_f|E7sYe#nw@XQ>%jN@Ot#3%e6w9VjVjIHiUc}wHmghqSjMhQ&?qD+$f>qWKXV2h*k=0vf_P43T9o8rzZLUDr!pF4SK4VE%8UKRi#oUs&>H7>8t?dOBCLfszB&2J53+#BXcZE#1zJ=(WH$@>wES{XGY<Cg*9PNdLV2!kvR8(28TIAXOWgi=}d1yXQg6Y8}rA2ou+DLg~rR9Qjw$6Sw`HOgtnnabx;Kq6W>|%L$$nWQMIcl6_!X^iOS~9HTb+(r;`$I<aTK%!l4!Nl#xWGmy3k?tRxGJr_J`2YS%Ym0?+X*UlZ<4aTg5MSrDC@Ga3O%TLllN>7p`GDV9X7F~sFP?1Ij4t+ebo9v6E4J)vDA*5H+;`n4LG7K#qpw%hTUmIZVKtoGd4(&Kc?KsNabK!x8_fd*u{EsxHbdhzm|-$vEWf4W+oFIS;+D`z!zlfr6+IMcqM8%0$tE7sjaVa7qpjfM_icUetzfj(ySXmye*WFVVvLn$o-=9%5&c0tOm68W#tf`FhO)3EmSgpN=kQ*>dNd)reDlh+I2`D9UEpxiEoQ$-L2(F+Jsi!f$UqqqX-fCWleCPqx6s)f9rT9$d9N9u{-QGOgcoB2hh!jgheDaVLx^~XUfUUEEiokw2TlCYebwUctWrA4X0@3QhKHUG<B5DX;IpcMKmmUqzwBh-0Zx!f1VunD@`3`T(<xH9s#_<iDF_4L}ZBauB~UQG6l@vk_i7uY7&&XOm~MKxLoqa`~fiKdIiw~lPASZ7}*OU&m?K`><H?h@E?MPw<u0kyK^qN;IG#oR($isdVbJe_hU0Gf&e7LDOnIm%B=?Fg5stR?4>;5>JC1(fX4(1mH80*=@l(!(2@9!}CxvS_e@3en0(0>5~nb4|zz!_N)Y#8g{N(%~`k&fvPI$b@9{jnNIdG8Pe6zcaJyut%qs^6O4tQObb|_!L8ytOdlJOn`YXutqCyB#mx?)y4#SjI*Vf3v^K;MLH5SSs4m7`Le7&qw>Eg#+Zf)jO-tkoadJQdh}|ci`9m}lb*|P^{n3`4j;)iW)3?`;o?H1G1xt#V^nR2k|pU3@yQi4f=aX$hRY>Gj`v^%)Ub}XW{)UDDsUbMrC3A0|B>M%2`g#{Bfh8Xm{I=5Fr!~_gWE~m6gR~3Ws3BLWTFFb<t$2@JM7A31!8G8x(*%1a@yx#CvC2Zq6owm2tja6hZ;yauR^Xnwl=thI%`GO3X*E373Q}W2Fd=P9&-f>3L6l$8hx(B441&lE$K|7=rrp=t0G*}@ySd$7!5+9f{q+U;ugy#>!C3b;MSg+kP+BV=%Fo2mu^}rH$X*Q**L@we7DLBTrWvjm!g8T@(Qw8j8H93WZY0U1TNL|%Zz2*D^AB^9bjMrf{g{{2;A(9G_w_#Rka$UMv~X7e<k#Vnh#)=J~EY!(;!o2o*A+TN*1+sjSd{zS*0Qq_S9w=CWW*wCQSx|{P?#c%<S%^A&Mw9bpM#Rp$Sx|jqbiRd&ueV-KYFV;s05K`}Du4VPij7t(sdzA{*k_CFBf=>GVZpf=l2BuPa_<asv@m#B^&byfYf2*!V2(R8}d6>bl-9b=sHsRji-P(x91h-07+GyH+}wS{~N>4bzSbwUx&n$aS@+Jy)e~53Wi^CAtl1rexu@ka5_)S0KXr?~WdtU#`7xW+fkCOCk$^y!GAP?T6s4Pk+)bLDtc^r2Cm(<!YVZ=i7J~koDBm|4fI-dZo5O#=aBn@*(|XE1Og(H5^!$@gtsR;n4TKcB_PQC~-Cid+LYx=-%#jJNYoloPPMKOyW&;{X&uDw2r6fP$S3G3RM?Lw#?xq-2WeSd0PP')).decode('utf-8'))

_v125FR_ITEMS = ('MELON', 'MILK', 'STRAWBERRY', 'WOOL')
_v125FR_STATE = {
    0: {"last_step": -1, "due_step": -1, "due": {}},
    1: {"last_step": -1, "due_step": -1, "due": {}},
}
_v125WEED_STATE = {0: {}, 1: {}}
_v125WEED_REPLAY_STEPS = 8
_v125SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}


def _v125get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _v125copy_action(action):
    action = copy.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in (action.get("hands") or [])],
        "market": [list(order) for order in (action.get("market") or [])],
    }


def _v125seat(obs):
    return 1 if int(_v125get(obs, "player", 0) or 0) == 1 else 0


def _v125farm(obs, seat):
    farms = list(_v125get(obs, "farms", []) or [])
    return farms[seat] if seat < len(farms) else {}


def _v125align_hands(action, obs):
    action = _v125copy_action(action)
    expected = len(_v125get(_v125farm(obs, _v125seat(obs)), "hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(order or ["PASS"]) for order in hands[:expected]]
    return action


def _v125tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (_v125get(farm, "tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError):
        return "LOCKED"


def _v125trace_actor_action(step, actor):
    trace = _V125_ACTIONS[min(max(int(step), 0), len(_V125_ACTIONS) - 1)] or {}
    if actor == "farmer":
        return list(trace.get("farmer") or ["PASS"])
    hands = trace.get("hands", []) or []
    return list(hands[actor] if actor < len(hands) else ["PASS"])


def _v125weed_repair_action(obs, action, step):
    action = _v125align_hands(action, obs)
    seat = _v125seat(obs)
    game = _v125WEED_STATE[seat]
    if step == 0 or step < int(game.get("last_step", -1)):
        game = {"last_step": step, "active": {}}
        _v125WEED_STATE[seat] = game
    game["last_step"] = step
    farm = _v125farm(obs, seat)
    positions = [_v125get(farm, "farmer"), *list(_v125get(farm, "hands", []) or [])]
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    active = game.setdefault("active", {})

    for actor, transaction in list(active.items()):
        index = 0 if actor == "farmer" else int(actor) + 1
        if index >= len(unit_actions):
            active.pop(actor, None)
            continue
        age = step - int(transaction["start"])
        if age == 1:
            unit_actions[index] = list(transaction["intended"])
        elif 2 <= age <= 1 + _v125WEED_REPLAY_STEPS:
            unit_actions[index] = _v125trace_actor_action(step - 1, actor)
        else:
            active.pop(actor, None)

    for index, (position, intended) in enumerate(zip(positions, unit_actions)):
        actor = "farmer" if index == 0 else index - 1
        if actor in active or not isinstance(intended, list) or not intended:
            continue
        if intended[0] not in ("BUILD_PASTURE", "PLANT"):
            continue
        tile = _v125tile_at(farm, position)
        if not isinstance(tile, dict) or tile.get("kind") != "WEED":
            continue
        active[actor] = {"start": step, "intended": list(intended)}
        unit_actions[index] = ["DIG"]

    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _v125align_hands(action, obs)


def _v125fr_state(obs, step):
    seat = _v125seat(obs)
    state = _v125FR_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "due_step": -1, "due": {}}
        _v125FR_STATE[seat] = state
    state["last_step"] = step
    if 0 <= int(state.get("due_step", -1)) < step:
        state["due_step"], state["due"] = -1, {}
    return state


def _v125town_demand_now(obs, item, step):
    demand = 1 if item != "FERTILIZER" and step % 24 == 0 else 0
    if step % 4 != 0:
        return demand
    town = _v125get(obs, "town", {}) or {}
    for shop in list(_v125get(town, "unlocked_shops", []) or []):
        products = _v125SHOP_PRODUCTS.get(shop, ())
        if item in products:
            demand += 2 if len(products) == 1 else 1
    return demand


def _v125future_quantity(step, item):
    future = step + 1
    if not 0 <= future < len(_V125_ACTIONS):
        return 0
    return sum(
        max(0, int(order[2]))
        for order in (_V125_ACTIONS[future].get("market") or [])
        if len(order) >= 3 and order[0] == "SELL" and order[1] == item
    )


def _v125pickup_reserve(action, item):
    reserve = 0
    for order in [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]:
        if isinstance(order, (list, tuple)) and len(order) >= 2 and order[0] == "PICKUP" and order[1] == item:
            try:
                reserve += max(0, int(order[2])) if len(order) >= 3 else 1
            except (TypeError, ValueError):
                reserve += 1
    return reserve


def _v125existing_sell(action, item):
    return sum(
        max(0, int(order[2]))
        for order in (action.get("market") or [])
        if len(order) >= 3 and order[0] == "SELL" and order[1] == item
    )


def _v125repay(action, state, step):
    if int(state.get("due_step", -1)) != step:
        return action
    due = {str(item): max(0, int(quantity)) for item, quantity in dict(state.get("due", {})).items()}
    action = _v125copy_action(action)
    market = []
    for raw in action.get("market") or []:
        order = list(raw)
        if len(order) >= 3 and order[0] == "SELL" and order[1] in due and due[order[1]] > 0:
            requested = max(0, int(order[2]))
            reduction = min(requested, due[order[1]])
            requested -= reduction
            due[order[1]] -= reduction
            if requested <= 0:
                continue
            order[2] = requested
        market.append(order)
    action["market"] = market[:10]
    state["due_step"], state["due"] = -1, {}
    return action


def _v125front_run(action, obs, state, step):
    if not _v125FR_ITEMS:
        return action
    private = _v125get(obs, "private", {}) or {}
    shed = _v125get(private, "shed", {}) or {}
    moved = {}
    action = _v125copy_action(action)
    for item in _v125FR_ITEMS:
        target = _v125future_quantity(step, item)
        if target <= 0 or _v125town_demand_now(obs, item, step) > 0:
            continue
        stock = max(0, int(_v125get(shed, item, 0) or 0))
        reserve = _v125pickup_reserve(action, item) + _v125existing_sell(action, item)
        quantity = min(target, max(0, stock - reserve))
        if quantity <= 0:
            continue
        market = [list(order) for order in (action.get("market") or [])]
        existing = next((order for order in market if len(order) >= 3 and order[0] == "SELL" and order[1] == item), None)
        if existing is not None:
            existing[2] = max(0, int(existing[2])) + quantity
        elif len(market) < 10:
            market.append(["SELL", item, quantity])
        else:
            continue
        action["market"] = market[:10]
        moved[item] = moved.get(item, 0) + quantity
    if moved:
        state["due_step"] = step + 1
        state["due"] = moved
    return action

_V126_ROUTE_STATE = {
    0: {"last_step": -1, "decided": False, "active": False},
    1: {"last_step": -1, "decided": False, "active": False},
}


def _v126_pet_yarn_compatible(obs):
    seat = _seat(obs)
    farm = _farm(obs, seat)
    private = _get(obs, "private", {}) or {}
    counts = {key: 0 for key in (
        "WHEAT", "MELON", "STRAWBERRY", "CARROT", "TOMATO",
        "COW", "SHEEP", "GOOSE", "PASTURE", "COOP", "WEED",
    )}
    for row in list(_get(farm, "tiles", []) or []):
        for tile in list(row or []):
            if not isinstance(tile, dict):
                continue
            for value in (tile.get("crop"), tile.get("animal"), tile.get("kind")):
                if value in counts:
                    counts[value] += 1
    shed = dict(_get(private, "shed", {}) or {})
    seeds = dict(_get(private, "seeds", {}) or {})
    inventories = list(_get(private, "inventories", []) or [])
    shops = list(_get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or [])
    return (
        shops[:2] == ["SMOOTHIE_SHOP", "SMOOTHIE_SHOP"]
        and len(list(_get(farm, "hands", []) or [])) == 0
        and len(list(_get(farm, "unlocked_quadrants", []) or [])) == 1
        and counts == {
            "WHEAT": 3, "MELON": 12, "STRAWBERRY": 4,
            "CARROT": 0, "TOMATO": 0, "COW": 4, "SHEEP": 2,
            "GOOSE": 0, "PASTURE": 6, "COOP": 0, "WEED": 0,
        }
        and len(inventories) == 1
        and not dict(inventories[0] or {})
        and int(seeds.get("WHEAT", 0) or 0) == 4
        and sum(max(0, int(value or 0)) for key, value in seeds.items() if key != "WHEAT") == 0
        and int(shed.get("WHEAT", 0) or 0) == 7
        and int(shed.get("FERTILIZER", 0) or 0) == 5
        and sum(
            max(0, int(value or 0))
            for key, value in shed.items()
            if key not in {"WHEAT", "FERTILIZER"}
        ) == 0
        and int(_get(farm, "money", 0) or 0) >= 297
    )


def _v126_route_active(obs, step):
    seat = _seat(obs)
    state = _V126_ROUTE_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "decided": False, "active": False}
        _V126_ROUTE_STATE[seat] = state
    state["last_step"] = step
    if step == 144 and not state.get("decided"):
        state["decided"] = True
        state["active"] = bool(_v126_pet_yarn_compatible(obs))
    return bool(state.get("active"))


def _v126_public_agent(obs):
    try:
        step = min(max(0, int(_v125get(obs, "step", 0) or 0)), len(_V125_ACTIONS) - 1)
        action = _v125weed_repair_action(obs, _v125copy_action(_V125_ACTIONS[step]), step)
        state = _v125fr_state(obs, step)
        action = _v125repay(action, state, step)
        action = _v125front_run(action, obs, state, step)
        return _v125align_hands(action, obs)
    except Exception:
        farm = _v125farm(obs, _v125seat(obs))
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in (_v125get(farm, "hands", []) or [])],
            "market": [],
        }



_V135_CARROT_SEED_STATE = {
    0: {"last_step": -1, "first_plant_seen": False, "active": None},
    1: {"last_step": -1, "first_plant_seen": False, "active": None},
}


def _v135_public_farm_shape(farm):
    counts = {
        "PASTURE": 0, "COOP": 0, "PLANT": 0, "WEED": 0,
        "WHEAT": 0, "CARROT": 0, "TOMATO": 0,
        "STRAWBERRY": 0, "MELON": 0,
        "COW": 0, "SHEEP": 0, "GOOSE": 0,
    }
    for row in list(_get(farm, "tiles", []) or []):
        for tile in list(row or []):
            if not isinstance(tile, dict):
                continue
            for field in ("kind", "crop", "animal"):
                value = str(tile.get(field, "") or "")
                if value in counts:
                    counts[value] += 1
    return (
        len(list(_get(farm, "hands", []) or [])),
        len(list(_get(farm, "unlocked_quadrants", []) or [])),
        *(counts[key] for key in sorted(counts)),
    )


def _v135_mirror_carrot_seed_jit(obs, action):
    """Keep one bootstrap carrot seed only against a visible mirror route."""
    seat = _seat(obs)
    step = int(_get(obs, "step", 0) or 0)
    state = _V135_CARROT_SEED_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "first_plant_seen": False, "active": None}
        _V135_CARROT_SEED_STATE[seat] = state
    state["last_step"] = step

    actor_orders = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
    has_carrot_plant = any(
        isinstance(order, list) and len(order) >= 2
        and order[:2] == ["PLANT", "CARROT"]
        for order in actor_orders
    )
    carrot_buys = [
        order for order in list(action.get("market") or [])
        if isinstance(order, list) and len(order) >= 3
        and order[:2] == ["BUY_SEED", "CARROT"]
    ]
    if state.get("active") is None and carrot_buys:
        farms = list(_get(obs, "farms", []) or [])
        if len(farms) >= 2:
            state["active"] = (
                _v135_public_farm_shape(farms[seat])
                == _v135_public_farm_shape(farms[1 - seat])
            )
        else:
            state["active"] = False
    if state.get("first_plant_seen") or not state.get("active"):
        if has_carrot_plant:
            state["first_plant_seen"] = True
        return action

    private = _get(obs, "private", {}) or {}
    seeds = _get(private, "seeds", {}) or {}
    stock = max(0, int(_get(seeds, "CARROT", 0) or 0))
    allowance = max(0, 1 - stock)
    market = list(action.get("market") or [])
    rewritten = []
    changed = False
    for raw_order in market:
        order = list(raw_order)
        if (
            len(order) >= 3 and order[:2] == ["BUY_SEED", "CARROT"]
            and isinstance(order[2], (int, float))
        ):
            quantity = max(0, int(order[2]))
            kept = min(quantity, allowance)
            allowance -= kept
            if kept != quantity:
                changed = True
            order = ["BUY_SEED", "CARROT", kept] if kept else ["PASS"]
        rewritten.append(order)
    if has_carrot_plant:
        state["first_plant_seen"] = True
    if not changed:
        return action
    copied = _copy_action(action)
    copied["market"] = rewritten
    return copied

def _v138_moon_agent(obs):
    step = int(_get(obs, "step", 0) or 0)
    if _v126_route_active(obs, step):
        action = _v126_public_agent(obs)
    else:
        action = _v125_v113_agent(obs)
    return _v135_mirror_carrot_seed_jit(obs, action)


def _kaggle_submission_entrypoint(obs):
    return agent(obs)


# V138 embeds frozen Soil V26H behind a one-action public-state selector.
_V138_SOIL_SOURCE = zlib.decompress(base64.b85decode('c-m~c+tPwcw<LQ1KgDFBE|suD<s5(~Jc}X%3SNjHsGxu%$Vq+n?dqO>-51*%g+OvtQI#WeRPkT`^<ScIU;lA^Q@{Qb_)%#66WihWA1l!h;Xf-D{l_;RR`AqA+xKc_U_UMIpV+d@e*){K+rj_(&woVgdaNCMFVt=Cf1kfv`osFi)k7<=^{btIPaXr`{pYMd&Nai2JoBH>dW7H8H$AYt@bSO?`A0fi{|Ej*_T!)L_rZ6qnsK!~llp(*zF)fey}-Ms|AgPaT)*-BW8LhBZ(9Fvr2p$b|Mg%0^`HNZs%gh4i~>v3{;}PgAB6wtp67@E`?AM>{`2oS;~O%D{Eq%7(yML!4zvIH{U2@4_q&`Xj=N11`-l1Gzw0<|=tcuENz!aucvHuZr~w-~K@cR;Y&K1!e#A)~gOMhJVmd}3ID(>0{QDvF@6Z4KkM`~0|L2CsdXqfr^@D+7I%yeboz#zpg&-DuL`j3F*I~G6){zEj8e{{j;|&5v8g&b)6XXFg8{~igVqA7rj_>xr_bmVCpYP|t4>kADQr>8Mi_=(nMe@As2dcc9)t|K9U5!_M50oO`s@Gnn)m^L}R;|kO!+{a?If=pR_*z_%52Nr_X}eoqg_!_9f!crnpL?&Pjm&Nf?bEUy+O5a7gEj{`?$7sqh}x&%(sz>LW?#MpOnq>WvJUq;qP!X#pjMv{bBcFgi@^cPd3d#bFk3)%tH-j{`rVzC>L*Bj!}xxW3-glUHu6DQ7={6Y@VWlGsseW}7!?}5`wW)vXt4~T(s8CaJj9mUm|}DWbgPn?&(3NmZ}KUUS81tp7$@rODBGY$Zq`I|@D#S2)-f4x@6_FDNzD^Io#5LxG0p9|MV$YIy5i^_9xqBfEibAOVQrXGvs|9<?-5RQ@Cse>E?%kEKU|$|_l(L0&KyKYEOwn|gXWG$pmWeB6h)rfSAUdt`$qpTbx5|P3x|67;QSh7s)xe9zigG&r8be#b2255^Qtrq5R7S;kD+Q0tmtw5d^Rg}mg-PJ7%eGj{7h@J>3KNdMmw>u4)k_DW$$RWi_>?cFnrwmXvr;>b4Zo0nDs9d_@eriR&rL})>V-THvteTv|rCR)3)Cxubwa#UKtRnI^>JZYOGB)B)KcY&2Ff?dWB%3_dk{mQ<Dee%-7hj0hW2%JT96|BHLU@0M)03%G=^s)mL>}P@|lpui)?6*6u17gf$?{&Nw<tFQ<x<DW_u*>JDW?zO*+lX>upk9TYlCgOB_D?oO-7ZpYMByL*0qM7q(PB*PQ@37RXTi^?N)&z*#Mh+xy>dphwzdG{}rF4c}QCl(j)Ytj<qvXoFCs@}$S%-vJ?QoQ+~506;2XaFcP`<<VNo!;mMy?&v(&ZH5Sy?Jy*?LA#L_xDt6XUv-{K})urdd176AUC^m;uW_&;!p_d)4XftUg6!n7dw(wlHC<_MaTWPb7hrI8=F9ew4IvXc<9RZo-6^2`@c||?ywqj4K!KeMTTqLF8Z8VOd+Q*yo9`4c+2WkiI=rOR*4Uo%QPgZ^59oEQ(|v!WhaHzaA*sgSN;o~nRlr&6$p>>ZJ0c#=qcYSCqi^6Nmxonqj(|6#R@6hhLEkC7Y5TsipyYFbV%bgUsnH~YnLn)XLD;Vjs>OXwM>TC^TS+aDSw-G_pRqpd1h!p0Cbd#lNKcseexV*nD`J6Fv79qWmXx=fulwzqPuW(@>*5bpIc>(^}S|M1)-59{0sG(pw<;KpY>ZMK3{408LDD5T1CWl3Az2_$l#%aj-*?OC6~q2@+W&hxOd)f1oa9)J&Strr$3?2k}FKtDML)eVQF7xk8t%hIqhzpT?1MJ|3b~~q8r>3bZ%Gp<4&15jSKzME<!0Qq~H{wooSH?IR(AceJX(#C-bmcX)WGX5$wEEMFb=F1Bd6mH@I6(DePRi|N5)BS;g3#nGl!fdoPMSq$Gm!c%I+KCo&VmfnJ-~@uQv^`@?G+xpJ#dj1Ztu?^Ao1bo$c;b&p%4O=HMGb@sA96-v!zZ#*2At=W+2Z!|_j%l|^z=Ob&c96PJxzSNX<marnRtvM)H(|Y7`)yLAKt{6X+H&x!AR%J)1Gz*f2ocQ$>OJ1ZZ_O|o5e`Pg5WP-6Q?}@^%83%X_Eckj~MVqH-5NA3D9c!ulOtfWh;fRKpqHIInC|6|AsjV87#qc(sl&&U^M3eoy8pnlW``v4AMWA1;_BzjKbiw%GDou!q3$EiyRJ%I=LP1A{u1&<~@)!~j7YhSm{HC-!PawnCh8dKDi*1AW3=%CKA*;B}Jx<au%*CrSA{#Eoj>`kA1x8vqaUZR7Y0LPR&SN(r*$m`oS|)6p6UjJNNN{{LpA$|;Sg^NzR!bp<b-JsmoDJqlnH#J-gZB}CPxkpJlMn=coxIDxP^DJl1Pxi8z1f@z!$YKvddBtIW#K%gLD~T8%08PL=duMtx3N0IyF<9BRlMa5*q;K!5%a@gw$Mk#(&W(D6|c{y+~fGGxgfBdpkLmiV*OIi0M&z-E7bBe*;p5VP-Z3-`*vPmfb`e!SAHOYGPP@RFzPUE^)a42CJU;wD-YsEVpy0Mv|pd-I5P7$kcdU5F`7RfoZ{h*OCtxZRe<=9AN30y`*h75fsWmjF_w?I+T~Xg8A3&k{}&2gMoyAnH=|K;vUtgJzIkr-D+0!?uZVsbFrm|)FQV%CVRX#fy&fvZc&bAXy}i$CgXya~W?+HM_B04V*XVvvFTJMwYlc^UZ8`f<D(`ckKkDlztfT{F$Ftmv;-2<9s0AL)9HX{v6Ysy|x|KQ27MU!YC0dr>wzQYbAydUav$6ts`Xh=X?Dof8%@hJwUd|U*TbxN)X?s#>6_?LqH0-qCL*OJa<R57sp$B5=F0W3X8L2r!p<q|}Zw~j0`(FR4Mo@fzmP-W}FccMx?xA$NcFr&aZLT2mFI3jq_ZrLTaw@~M^?Y-vR}6aELq1vs47|?n|Is$<+q&3#ZMGNt@R{1y_1#e%aVFEtWs0*daX@YIg1tnY7{a97n|3ByIc&(Q4YbPyZ5;%|(HmbAElb?V-1y!qtr8|m1we$806X9JGx=Yr0?0%FZFOfwt()-LYqC>Hq$~o98!+L4bNfU#&Er>|l#8d*QoIdO-rSo0JPEnvQYoWcVOiqWq&TTT2(BQ;_)=-E=t6R?fLqayuft!Hq(~*2+%z5!Mx`nI+%;h50*`ZV?Cx&NEhz5t*{DG|kYx>4p@$w0z0RsOJpoK_+V!mM-ty*0uaNM|1zueBf|U_b;jJe8(U2B|{MyGOOv889;S<V&aK(;9&my#ywmh~q6GRG3)oa~*wal1Q+RfmAPoHWl)^<yf%1gJ!gGIY0vO8F_$|PLiomQovb;Ao-I=g1ZcDw&V-G=4oZai(g-M&20@%<G?I@mSY8M_bIDTF)feWT>L3Ut7AzyGC^PCKk~b*#c0+)_)X&D-GKp0c(P7VK^|=j3Z;Gmm4!m~zt0f@W!-NX>tt#E-fpyeHq2a2@JVv$tv^`X(JdSH1IaP6>yx3?3nA$gIm<@^v_>;`2)`n>RY2k}O#|e9W~qaIF|26$G7dvL2P*-Par3o3}~V^9qN;a`P|LL<{GqYR~n?LD{2OGTwWej9%T><f{AYWY~V&?okM!XSAk#zMDAm?)^|Q_q_tv(i_Q@almD*?WL6Z$3Ys#{xE<?!z|Ut*F*J_w=>y^6X!J$7if}|BM_+W=EWq}z0X&jvDNn7EBp(EEy!}JIPxlpWZ=0syb;mJGgmA^@SX#DcAfJUd??0?$}{(&$X}C4pDEWqD>?GGSa~Y*n#KI$Mn=LKduZk83Ro?9)P9T&-mHGpr1<IK@JNIzDZa!NRyXq)hLO}*Pf=RZ!X+(6(-`VZFiLg1NmJRn+lwyqLefqLt#@dQLGXHj9xIzJ{tI=-2c<!ycvRHtK>~x8b3XwW*GUgdrs7FE+JTIe-aHk~<P#>d@IYm<Yni$UV6q&}ZhC97I97mD@md6=_vo=~@5nztb$ee&YuZOa?~ycho+XRi!)l4kSw-=!PYEErDvve<<ad3zpWcRcn|d+FRfj(JpLRzAhtDS4PatzNte4aox_WGR0nyiSJjWJo5A2mA0@QD_!6XemPUH@lK58r1L&4#h%~~#2nH9W=-gt9>R_RvfNBy;8JUBaXSb5UeKsw@_UqF4oc<S9(&-<oHt(BwpCfXsOs=dsD%H^zATn(3T9oHB|ezZ3cG$jxMrwi$b_eqp*M;d!|z}%|V(XDN&P`LUzIx+L<0*>=fD=x&9Jz&zX*5uaKI8vggb9VIMYEqh!rzw><g8A!=&i<y^>n}+MTJ#=_ox|;blZ++|D}$#5=V7x@QH4R*onkF+jJ|f2$fV}|TN@aa=DMc1<R!Vy8!gh@>?4J=^TzuOqQMT3rXy$z%*Y4VSpONp>Ur8DxEh=fJ`gXxmu79oeX6wGUSPG`+T1O|);L?vNAnl^F$xD4#uSGo<ldGY;FYB?bgS@$=t`T;>8nK8P#c>~BcZRqw)9sb-QU#fO<7;<&bQCHK(o_oR1(pF+yts`y$fFSrdkubx1Cf%hZ?~>25Q_&J?2^}&99@AURF0(xwce@c^UA*LrdSe4J{)~agZZ40(if+0PzNBPtQgFnO&uE2vz_m?UqVbeY|jHdrc>++rg<t$K19%yC)TvxDQaRq^z2Sm>AVYn!59ya;l!)IpXYRFRxd{jn?IFX5Q^GM8<X2Ikhf*iyPbQ;LIc&^-#kREvrJ&=67Z3Hgm_R>dbZ%AY34WNIXn#7iTM%2Hq#0NyaCo+w?u2boPxGX9=@GYg7olDfbyG?D$h}2a{fhYsb@t5U>GasrzR5+8QjkKbi1489XnW{-2Ja*76VcNmLpNCf#K7(wD{)MM*`*71r>;UgDd%J0&IfGW+^>SSNWb5!w%ita@9&^W<-XrrkdQ=g<Rbd7Of0dVdJA%ek2BCq6ZuJx=X1QAAoJ?dQYljbeC#o8l64AGhKRZXdU_yWPB#=+<dMp)lj>)XFI;8+L@>Wc6}wFI$g73wl|cuwXx<^5d$8`&k{k=&ws!hJ(ACKIuUTJ&g~KMyt&lKL@2ItD+E1DyXP9{G<j)k9Eh6m}AwcZh_PyPZKa+0V8#zk4`=~H)`$$M`=PY2ETp*yD#l_%UHYJ1<@EAmG5>TxZ612n$D^(_>K+RMgxzjai?s%3=WMqeDFTA?!dS{CpU@c6O*HUX%3%>s3)#GY4G&=S(jb;_^5R6*&d4x{HOh}#h^1-e|A%CQR}rB<~q$af2!!^R)c=`@1nS^%_?NnW4?MG0h8-k-7JSu55d!G7XMj*AUAbUw45C?D%UMXZalY+TRk8tpQ-X91cSR~_|a=~r1L}<$gc)JYr;GlPzF}E=v&N!NPn?zZjJyCt2guAuVzkOnz_%);@V!fk4mikW=-!;$*s0C^-!9`q-=x^ZX5OrLb*tikGB!UiWkvu-EW@=KZJ&qqn)WNv%A5iv^HJ}7>z3p8F2+(Ic(hox15u@90b51fE2v~Jdx(t0VZ5o9Q-)AjkG&6PHiN0wZB*m{cG%$UTcAhFn~x!hizWHa@kHR{Z-S1zaD8*)XO_H+GU{B#H**?si$zPn{F~?C&%0Y&D)l?YG07+M#3@`o*fp8pYIsujaB>o>GXMDoF)_EHHs_NB+B-@4qPgd)$PtVcC+6rUP^3raUa^d8E9AGEYjwifaDahi}#|2eqdo!{*0gM&AI7|LG+!E?Qf!*k8=9)r?$>rNo6y6C=14@^od<`>;U)S6f2CiZY0&BgAFh&J?M0YLpoQZzSiK%K}Bn8UU=vfmdj&l9zEezn!x#S%cQOY21G%ZvX>R{e9~oh^B-^59Yt<wh)nv6p!-0LW_`nR+Y?;!vem083oS42m5!;*yR$6@2G&KgviOuga?Lw|<5|u!nY^KNWt<JP#c4CX%dNCs+_sD*Hg6^kw%Ywo^{n@yo=>&uVN-lMUH9lx!)8#&0J|t_$MbHq1oJwm9m(aU89@%%Q$Kcj(v;@BRcE=nfO*GBf2-A82@>YXws!$`A%Cw>YP45(a%Vn6efnRhx`B<lDQ(f~`=N7tV#9f~YgU}3ZL}n4wMf4mG?+L>)$Gq$3Qa8T*nXmwoo~mB{+4Bz%zTH;w|Bfx3H|wEAAU@6C@=m9+z)2E>0RvaOYglYdvDdEACqj}V=7+*C)e&SFR;41J}OjJYCvC6G+<>{75g*0y6}~5U*7J}UDZD>0SUQ?C(OKYQ`db!o5}QaPD??t`g1CQ-5O7c&w4ksgO0Up0JG94iu+WncU_kuC7hb|>Suj~f@(fmh<RgSj@EWboRS<}re~%aAnDL_UFB>|Dto~fHw?5h$2(Eakw3PF=s=JncxpbJjTzCiE1WaLe7EQ?Y?*R94l8CEwj55)4Z_@BATCfQ(0X+IUKq~7+d|9=PBBpDiao6}EHjd_la+Rga79(octpWvz0L+dg)5%v=BySSIiC1t#nmUDckuImKi)jmu$>Vd(wy}MnL?MXXYZ5N#==mBIdj-~YG1~FxCUch#ltV~{ZOW!w}(`e&i5qA2Xezv1gl>v=lbgg<J}AE#pa={SySPl)PhRZ2bVcnu)3|QMJ>(taFH6u`4O6Og=ytft%jU!L2Z1RBc&F365TzXN54uZevM6I7zrDtgYsVi3&$<Ie<>hw@miQ4?*7@UU(0**JVd3>Ebr-r3z4DJWXC~^GoN&Hvfnz<T64{$J%!;Q+>2ve|B^GcmNZ`I*5Bzb6JxJiCvt`y;rHvsUHAAYM)vgK;hVs__knw>XWyy_tUSw-WX6=KP!!V;23FXO3w5R_b&ckQEd(R$-S86g+)o&|Ed+H!Z?xQVq7LCY)rEgh^p2c;Qph4>?H#K<wRJiz`X#4)FO$n>XS{Og*Y1_x%6E2-kIwNu4u_0KR>gO><rP@Kig1*?ygJ3hrPDSCeMiSzpYG6`7I*mkUa+{x{)uI^*N)D~moVM+L^$}YLD{%6@CrH{URN*A6Nz#MVzTyeuGhJ<H<s$?8fT<ZvMLlA?|NxiVKQA76uZ5>o%dUC^*R)3qc{1^HaWp2H%82!|3ZyH2uTc?CcCFyospKryu_AHxY)nPLRcAN-<jJU^2}wTlGSDhgBH(T*G*piPw~-ppzUlJ1CotO6xwb<&)xY!mPeb}0K0)=OUaNzwXlZvKT+DW=|iKwkpm%dZl|iTsm$nmM=4tGx7eD2NTL$=)3i^{i8zRvN~O14tT(q%oO$3X=MXLgPNm7P%`?P#SKjM-@gb?z=1>_~3P7Xo1q^>gJ<WkqzJI@~9YbzkABj9h9{CCB|0YADkh;grLKe%7w5=o${!?kwql(F*!TA=y-$gcT9HHuFV*<MqG_t{mQ9hf5F?^&may-69rEEL#uU>C6S$l6$odFhU>ete0Q=D@ReA$nm<gf1VYEQ3I1Bmx}wNR?g<Z5fWKW%E8<>g#IW6v{1^>MbO?m=}Dhby@c#~+j0de0Kr6&9mHD=pgRlW12ZXWk_<vA{c1D!c#*l`40ryI$34;$Ns!*?_A)TP>cyqcq&=-R@VmSjwJyI4`Y72D?*>xS;);e;+v9+i#^@>1)F#-n=lnl@#@^R7!ehe#F~{0`?KWQsY3DN*;2k*GZVKQUo#x8vjB?3nV~iwE{?s=B|Jk)VZ{_ebsALMP+CAu}2{|8hPFW-PbdkAFhvgDWcSSCIvO!b9^QMW?|5eOONa@Jk2wy+B-kn{pe3TI#mx|5d9tN?S0v9+&BH%pyypNyih%D+pXo_gREfbB9+OU>B=wDpCu>0Jw1j+{_Mx~QG#ukpPrUJ&_Ynk^9B^$O{2@sE(aSlnL}26mq~^{#k<b7mt7e4aZyeZ@1<ivy>~RUK3&g7{NZ0Hf^$^$P-w`of^Un~p)nf2XXQ7{4R)LS>cyETzky3~nD2w=@-&f<>kL8GTQAb%8Y(u=tok&s^J3B+E~d+Eqy~SZu{2!yWp-V?SHg@~LL>AbHH!B<*b;}EQ`ygnfSR3VjjmN5RMDDuOb_KgwrY4I82b>533a$&RD00uTNU_O4fq2nqSD%k{tQYR+3bXtIZJfhbM;QOzg|~&(ladHyPz(9?gFyiW3-HH0P(#)IXO@r8Jyb9h3lC_Ic42yUKD?(inVZ*_J+!)(CXa_lY!XN<b8;Bh0dD#Y5wB-9Le*f+zyVnK5W93y1h<;J`PY@Ujf)?H(VdLg;r-fesSe~Txm7*&dw4AixV*cQ}W$)LOIJzG;Ke+x_gm}>0*>S#}>E;Z;<Uj`RteM)e*1osn#j6$mQ44<!vo}_F<V@0Quy4n7fRJCQt>)VdO$j5WJN2zhm7w7iZ?Ym{OGwQgR`^=j`%fdG|=3{R^~(o=&M>JuDkUF&bpI>#c|mwwHeF;ov9}p&$HnqePoewKQO$9;9_gvY~(A2rFgWre;jCM7^(&xrl<3TvuXY-Mik4B4#I@?i$~2XWZr$nCaMiuSfj-Z!mW<e~aiHs;RAwwN@`@J+SxePfQO_wv2HG-p`df?n2w?1C=-W?pk)9>!$jmAmM1}U4%PAd1D?;`Sf!-4?C|Of#b&+aM0%Lsi>5^z?9zZ&q*>yiLbs7i;-7BZ({&YdhX=ZlN;yT@XKz^&cUcs1$Ts0zzVK=gjg}xqvOm(KG%MA^1gO>tCXqO5H=XEgPG-wvxi{nbvoUnd^8l8DV%;4?G461`a<;9wj+X-{X^|DxNc+ybH4n%<!6mi`ELov>r4i)`zYEomW&?{kW=ic#_#r-cX-es>*f#OgbY`JucycRMkvZsv)C6M36vfU<WQ~%UtK2nkl4V90&~l=b<b4YhRRLb;j*;F@r9E83)RNx(N4+chr#%bhK;g0sIS4ujvwnm$O>+0(YmzwIjncb(zoZFr(!HCY+pg5!7(~$&^l!5*WwbNHC;twCJ3L7L2I>d&5av&D!!J_#;;A?w0++UxA=o7X|6K9HlU9($mJ1ODs5Ehnw{?V)0lWuY0mHD2L*@qym{ElVm)3h552exd?n%eeSEC|`J6V3qovU7e+9^$5KbP*koA3g`&<30B6VOhd+(_AvzImHXN-)%S^(mAur_aX#e#O5@+T<h!rvSX33XI6rG~^|3vXDPqj40LiwnTJeU)9E|M2_Ot!=W0H60K3+rKg?mF$g8ak`!)c2By`GTK!-l_zP?`x`alA66GVlQJGdgYi{OajJkSi-p0$#(HcGl7>%_2RZ2c?Nrc3o1sQPDFbYUaox0D9cd9Z5*54)jI?;Sf~~2t4&i*?y-d@}wG5i%-<mXVz{|*DqBL%i)w{`Y8ZA@d003-3k4QCb5yQ4pw?4Z^pCy{d`yE#!=-prc*={OAw$bj^T<$NyxZPG;4<)TI%zfu+c>B?^&Z#+jaO&jw{HWP}PX{gJB(FDnCtFU@sZc8V>w(k1aQ)W(9sKQG;7)0<_Iq+sLmJqu(oKA8_v)S<-<V(h{pC?qG3tKPhWw&kZ&kP}<$k(GnQu);<9=ngQL4C3G_sC&A2Iu(L-_F&Dn81bN3S>E!4<DdaJly{l+e{_B>?Tq!HC?ow_tW3RZU4VhCi+l8Ba>9KCB*p&8qWf>^KzAP0?OKTUEHNP*q=jU^72-*Z1%|{{>&|#-6OELC19^Ukx<OvD=%hW2F32pUmyAfGjJkrBH_+?rmKtFaHL)-^3qmxVHxreY|KD3-|qQjNdQz<9J1>#uFwotASmYScThJ0@VFflnWO7<8M!O!QLyb3uzLy&h4PG5-0a&n;W*uu}t`kN$}1dP{-wg{?n_IRIHaxSIQZrd0&sy#s#WEZ*%!|w{`1rX%Fgo#G~h}mS+s{DowKWrE%P4lVrhO{R&AH{d2Ge15Da1DRUCtt+>FN`pTLtp;uo6&Ud6b{y{A+zRK4Y-9y8=FxcA`zM2z1w2i=X-#Z{*jUSVGV%91r(_V3#GxqQs6Sj3`OVv7Eq%VK{x-7OlVIhp3?eJc_=FJ5(i0R%fgvz-(aj4137-F-*IP#{PtWI?_eieWd+i#%RqaV7h&CO57T}0zXZoF;9M1w3(6v2c~m{rwQxY~R+{wOs6lm}bipTmdZDB$WnUVC)r+N!)CexG2p@cHGYQ}(i@me3^>-ey6E(i+yx;roYz8<o*Zy4*0~b&cNI1UdOzTV46U>GkLz`<n(w?LzR}lz^r=)dsWMs@31WmxJbY+2TugWJ*@+%L>BNDuP`1Hl=dcnf2w~VR#jD(9uTb{2};9AY~s_3{25trQJF09*dv#$<0Dh3wSl?S>NcRSG=%pkhFWnOh3L=n)OoQ3Q%TnfqGAgkO(lmpZaFKrx|j^z4<(6`}0Y^BXmmKO_;Ec`>g=T7XEyXgrx2?AiOQ>(!Wr~zIJb-FY8hjPRV>_79fGC*U~5WkVc)7#kmKUe=3*yMNFET9orhBxyK;m0_XpYdvuJ^220bzs`Sd*BXGOlo+#Ig%-&oH?IEz_&`DVJt4cTb9KAizrNLSFdg!}^?uQ{Ew!IB@O$xmdd%A-r2CL=l+Qrv1GemCIt6MjEKXN}{L<h<aG!?Jb`^DSIA^OwjPE6JS^GK|$VefeK{GQE}lu}J-DbZ*%E*dYelZQ<!Gu6IsIfKq*&%Gw6+@s3gDF!DsoB_EOp1ObK4IAjYHMRiDRg0NfJYfXrX3P;3z!zHiK18{BbvhhMP!FgN=4w4&7o%b&T;9LRI+<LF9-I)cSRHvaG}oXoVnB=jqujfu*a63w$?D=2=37c*!R4%YY&7H^`e=|096l@0tvUrd*YF9o=1RY4x&ReGxixN`9-JfVOgqg&n>-M~sC(4p|GP|}RworY56zR}^5yX)7uq$1e#@U%PppKSB@r*L52Z5?_8(srgv7rMOIZiIIrii+oE5M0zfCvQx12DiJia+RWmS6*F)WtA{;~sJBOy7rPTQg}iJd&b*!bQ)ZjUpYIHym2{C_oE&h7gVJ@>ZYZE?AyTiAcbD%s26`Gr-thGv2qjneOH4!TqBEwero4!Th3MDw4i%fVBhQ`63Mxm3h`wO4^SJ!%%W!QQdu$ZRc<()|4|6u&b30_OFLw<391Ir-7Ml2>y$CQ<otXz!DAV;2nA;hYe$;9e;3vst*^e+;fHycPBG?euE&aN^Fn6S*q^2)vzM0B6%{X7)~lJqPre&ItXfT?v*A^Oz!x<R|yLcoDrqd0*B$?b&xk7B+m0y}Hdg^d<tl$XDJ}eLEcAT4Z^w%t|nU&7Wa3eA)+J8@#|xj|10Pm!Se=CQ41}V&0pliuZGLr|bHzWU*RnsCl8ksoeJI!eO8G09ddGr5<VDDMM6x%bH=AZsWIZ--hquwpFzG`EU(6eYaOHU7v!tdQGnt9E2504UwQY9q8gQd5RY;`&o0PsrTD@!ELcRp=EO4)NHA@KosuI4dLYKY^3Ah(uk1rt(QjC0%6rU8kpDg7U=gCu7Y)K?ReU3sW#9{o5%K*uO4Px&1Cz*mez;wsI}m@pIV{F@cp)^tiqjJ#kcw!E^p_G5Av4(C=^(Ca@=xazFZC)bmgs-19%_h9AADs7pZtu-f4lwM+&#G)`3dyUz~Q&ABwp}ZVB`{9aqI0etPOUjE2m^?6zN@873VvQ2i~7dYia-Xhx@H(?l9IP|e#5yR>S<2&;YC-fjBcU;3m*mp`0<)H<(_c)T8Q-&YA)zw~4)$kW$U=?jf@-}j-voq^}D)l4k>v*KDRzD;)1mWQYj^IS6P!4>l;`~Xg;Lu<cW@x6jl*B(9cT3wGyIi0=Qub|osCMo0H+0m^ZZX=<XU+5Ba8q8Cp;FPac^Hlwn9A*ICNooi)S&E94FFQ|+f6|7XUWkP|=MPC{FQ>#@;->YPr2K72sN(!;Z9x6X;8yCWSNZ;&$BS?@E>#4VqnAy5bu(KCQ`mw~<GF$QM%Zx=L*3?%;nWV_f*Q>id-gt=*BJoPPLJ1^i_WR7fbFDl$?qNItpCdBuA%UAvlGBu6Iz>BtavbzYpORJt>N5B*WKPtZ=qw#Ltcv%8L0D5b@l9yXHKiqY5!TEfiDW^!dlLcc`9(c1806GI-X`jZ32W~Z`RRgQ@Dh+I##?GR|#^nS+B&|iCk}LOD&zxHcte2ZhCG7UywdiDhizmWOJjsN!33@Yj<AletZ7-Q@>Q}?)kgzjm!)1H6=-KN={)5oZJshc{PmEa<oh5!^4mi4!prm=tjI>O+ZpnCRl~WaHl?jt?|uWIm<m9((qw)n%bH%`Zea%K6XO?0pR_7pR;Xmmrtj4W=e$Q1_AEEFs?0@AKxf^&G6M9_jPuUY<nS_$IbWWy>k8Y9g+{tsj?B<MppfjC%Og7*WGZ7hCKmpkocqu{t6F^8GOf7%+~0ZK1=1b%^jksuU9@-RZ-q6$$0^zS8oa?Mg#`+ziNJ}C95(!kB*ruyo6Toz23JF7M`c~`>f;eI8v_zx6g%+Dn$ZPCKmUh3kzyHJt~jSS7pRw5S`QbK0jT`U;}GkuJiZf{7uYp$nGD^`{V9^ay_BECK@0jqe10s$~)(gff6oQz>txd)w&;(&5cF<u3-M^x?3OX>mzdJmzVJr>mft@*>0bDXtVfe{zC7~;C}23K$aiqy}K0V%i*j<EkAQY3f#5zTAY#vH)-B_mHaXvSSJPSYp=O6%4G6W9U*fKKK6p~ZMg!+nsFPLv+ch8hG*_%Hm0#FGA%)$-wIo0*j}KkZPU|7yFPy&ZwaBB*7G~mdo4J4pP6y-B7E+rTVAYm1e)D_c*{((&Zg_Pz4MuB8H9}C)N8&&>;blTbGi5E=0J|xy%T)ryIh;D{0mjyZ5KUpC;QKtf<H+`o6O4a2&pdv8fjG@VS9HP1iN|;z|?tqXKYuH_geAPXYl&wNQ<S+?2B3VT-{8<b#&yCZt>`_n-|fiZ+<UUyX9B@tCoIeP;KFt?6~nc_I!|#M{a={ZVa|VzZWWSn<n@tUwpl5@NDXX+^y>5Yq7mB$=|C3n50oHmkXrIXub2=gv8i*=UMrXGSI@`%}!@ApqJ%@4SKJ6F5EAp51SMzMW1Ft6Wh(smltsRMpOAeFLV6=ym@hRugbD1<+#k)EsQyGl^DPE0;d)rCcFNeWz$$(d3egNo~^%f-=a*nPjrvzUo?2utjf{;H50%pzBo%w8kygw<y+DKJ*-~5c9o`V1h;YPUj3lTT#GBvRve4O?^Z?nYoSAc?+|QJ@coXo9MC7Kq#6X<v&{-|I#~4t*~Q&J>(#QR@5jG?ezKskMu;IatZP7{gD}HEIH^2T@O~1G$Ixw-&>=pcGBPp7z3u<({(5=q6TJjTrvPp9CafKy7HZ(Vn9<_u-5vAaL9s@6e0|ye^n3T~;ty(D)G<?7t>dMfr0r(C_Dw{+HLrcb5Wf4{2QWSKq^b9;?^a4{%i-MnG#Up*xw%(?`;uWgOgMz)EB8E-JHgsP3#B8Y{MWE7M+br8#<jb7`Yp<aA9sZfhfr@+*x)kqp-?BY-psbdyZd%l69!r8chJ1JTg_ubzigfyZz&0W45ic9i*BCCkj1|hxq1*w<7GWvlh)YP%C=WFzvIM-QE&R_nBi^lNuKy30lv!7e0Q$uV3us%Pjhg3I)0&Y=HNHhFvp{+UV;4xbRHNWlrn@m48{0bb~c-aw}5_WTJJ!DRyn^|_ZZ2m>m;z3-TeS^_LQJ3PPdA{r~G)zo~lKBj@5-ep0ctayKd{Fm-6hzo?O<(Ulriug@WsyKwKOy(peqJ#IYmsD-(N9ei?T@b!*)zF|%5m?s3+>PYyTu{!nUpHl7gm2yqReN^JEFAPuWylPaB`6iXeY7G0=}m@liLl`<D~<;uKvbk)|fJ!JEm_-7(8sMBRP`ORr}wmosiRZA6^Oe)=}XPba?5gDo5G58rf>&4-7PcXA=m5bb@Z3Td%EI;IZg!js&hRTggK{YQxRUa;wseJ|%*z=*pK8Z|k{|oiF6s{BJsZL4H7*ASO*2(Z29C9`1$#+q_BeR7>+)jHs<6ojDG@viz!|eNnR=v?gUN`&C$Gi7tO$w}^>_y)mkuTjoXgr@2guHws`+XJrt(XE+UHGeFzqb>YqUcX_I90PNx=p(a%`W=3ULTZ^ktAI4<Grrc%e!_Dc(>o90#P2-Cn!1OW@t9*JzP-RLcCEtOSJ1}>-PdC>PcE6dW%D^B{`jbT>aH%DTY)Z?rMIa$HU{aGWtAlf^9w<U47~LQZ4CP3x3Lm*F#a&<jJDEtjX)!ovBdy>HP}atd@jS-t4p8%A5E)N*qs3sUQJgDqUk_djS+@-JjfOI)%YUD~@<(eiZQ%`GJ$+(|wd$49ZkyWyTxL?s2;X&7PdQ=t5@Le|G@4KX4~QJn3$U2d{dVe9;-HH(Z=8wS9{K4_ud?>Dbk0=($kjN<T&zNxep8Oji~;v&k$vz*tP!eO)dz#umXeoIfPjIyrxDC<I&&9vJl#LdvcgTv3f2E1ie#`9ZRQI2h(|`E}1;tuyI2mJnh@-g0gSt=X<8@ddglF^_axC*Jt1p%xPiq?*t2d{rG(<KUQFkK1}xLVvqIMPpoH?KM2=5V<cNpcp-Jdtck(+v<W1b4ODuqIk444WtA3?@hKL+N>`JFE5%z&%8U&3VZeTXy2}f3Ex`Ulf{wCh=pDOT06&BkG!v1HK}y5g^jk|VnnR=XE$C;lO&fndpfDq<FiMQWT(VV3rd@N$g(Dwn6~hDN^!zn-E(puz@T)32OqI;DeRhdYdp#?pH9TD%cZzY^F!!4H!2^oLI+zg-8No&NR27=*HU|BSXilJ_fG+QoJvC%b%wC$_h+$GQs<XSyp?)wIglQ;U9BaS-M=ZGo=X@B@?i++<9YeCwCnTQ)-6lZmpYf)RK0$2@ikS4maoBniPBg<L|b}R^NWfLKWbdJR{>Drr8X~ji$mU&PhJ}9?U@A&Px8>Koi5a|68;L>`z*GbZaRUY!*FFLfOQ0y02T`|8K%d1;(znKyq$WkkI<soX^qH=e7~GBoJn5UW>cKv0Lp~W-$)`huh~iy>+TA9h`>4+cd^g#?QBcQeI7JvDWO<pv=8>BdFl)dmMO$LF%gJ<F_=8-Y^w@6=%*cE)lYS+8UVLmlUKT>l}E%A6YuTOWR$mm)vJ_SY!A;+zcq4nJBIN|_r4#D_tu7-FLv$2y*lsE%{DWf);X<+NK6HN?u=m;JPWnU7Oi+1`i?qU3W-m}#sl85@2Yp2uM~nD4x%#9dHub(22UceJ)ew>4PQW7_hWKgTlCnSHWf+QFbp|Z)eCs>77TKKXg5+Gxxc#WTM?U1vx0pR5{K;t3+POz645l-?TzQH;t9iVo3%|J$aBL{_vvpYR+`q>o3BPp@h}kh5?38xfccQQ_R`bwB@T&Xl2La~EH_F_bB8XWa$^is5NJC*ryC#-+c_Fd2Vn6sqxSpm*HD+IQ0sB+e6zMx>iSrus6VKqI`8&QbYaRFtMUFu=w{(I96OWs=D6{Lrw27+BU;qx{r!1LL}yIwTVFPv0m**doYl}Rp^w4Ea5Sdg*<bb5wC4TUTp_Piv1<B{6u+j-bc>__lH9uO)7E4{dRvoM&-om*1kMiSMgh$1i|g?Gre_pJFqKyiB%S>&-F8UqQI_2kB?Kl-vQWYb(raJ7gU*^yfpj{ZZQn*(78Iokj*bl5eojLD+G#JI7&xjo7Hup$wJPoD0#1?L^|AwPCu7IDp3H^{QKe8z_!ZXM&Dn`q^~}r6@6pd}{zP8jrTH4o#`xhgTT`3u5NuH!PO9BPBK@B^Z$|-4s)K4i($maU?ss6~jrzMTUa+;d4t@^RI^Aee$zwnbfjX!hqJN=cqZ>J=X)Xz-dngo}rE`A?7b_2TG98AXBymi`QR-_q83^(7Q8WWO2|mW|K{xIPYXcroA>WUc3+@H^SYO|ct!qZz?DN4$3#F@FDR6@O`fUWc@Xf|{jB{qABjM(=qYvcZ-hY9Qb%B7Y@lkO`De7y^wb*@|4@&w<@%par<0S>%hsAEGt@@KSENpUo9MtdO@>GRG*zfO_B)VukPOI`?DI}?1mrWB6FGm_2nag!)<qejru^RcyveEKEcW__2DNsJ#Rvq|WC}y4KH;##+vl|b9<isMnGpZf<urA%oW%ROGpjEj~e@W;D!2?>aJKNtz&PlDV9PQMUp`uezYkq@<u8KY^Akea#!fK~hNu;PO9;c)5l?3YMrc4S6x2MW}q_^91J*^mna&Jr4?Hb+O)i^Zs3nOkG(SBt?X+5V5{~Gl$I~E{WoD@ifO&-ryh~9kPO7$01S)Q}#u$h33Cnp^TB}Ovfxiw;T8h42{`m9<ZP4HfI3gAuelOv`3A!gSFxM^+O%HebL=BI+&n=Gc>PdGNiCL&9-CB&SLO%{KeeAQQ2=-fgouUw$%q{?|38<7X<FD(l5@eqQS+txrM7W3u$(0G8Jq74pM#=f8U1;yg9IbLP3JHFrLb#cg<Ryy2aP17*Tk!X7ljlW`bGaV2m;eZZ0*|{^(uA4gc%ARvw@;1M{c>70k_qgg!jQZ%TQGtim&G|50xYOLGfp&K3>Cfheo;sc571*>={V7tJX*}Z((@e$_1)VAK%=Xp?eU*xOdyxlvI3pqM&$Ov!CEA$Ax{gsdVz{2(b|z+YtItwHD0ianco;nYO4Pe`nH=A7t_uuOcOpBd*_STsFgdN@r!~u;cZ@?oq$k^G%fD6{XF!BsA;MHslAa?)6ylrv!4mhODqWXtLwiVXcB4I_l9W4!x2D7{!2EVOwD*_4QrfkumY(i-CM3AJrO~Y@^rGPN)w4Dg7Sb~5pljkxG|9YjL#0ZimrGS3+Prt<ve8O3Vc@KQ>L@szgfHu_&60bO8i*gO;-g+FANqZNE!aQpP=)}@8k7gUA;`($s(>S;;A1zsHaj>Cf6#1g5Vkg7UzAKwpzWo5HgYXi<jyZtv`o2mhs*g>sS{UD?eTm%G|1{{SnnaQ%5NBUI~Z_1+wt68A?(>DA*|l*WwZWxZ6%UAi$lQ+M%(93ZWSI5@cQH9_iox+cQ~=WeV=0z%lY?oJiF5DZ8k-X<!WkIchs5j;N!T$ubM36`%qI{bf$|{d;`C7X8Gapd*lB8TR=j>BgRjc<X3)AO6}Huzpc#pap;Z4jl;MagH5k6FImlQUfKb*#dI%j+v^gkv#Z^cGTh;Y3ji!6f47@g8^QVbh!+0#dXk29(7S1~ti4Z`&&4K-;C{08dEcm-<|t(<<z0q+ZTNbWUS7<4Mb&#%@!eLzu*%e%{H?U&7O-eM`b1X7&+S?1P>8q9=HDB`AK=h--L-m3<xTkbm^}76iR*`SA*t*YUD4N`>s#2Ge<dC@1FDs6XLTg5&wjict5rxz-lZ9a`GQe6G&xUKj*e1<>$lD1q@3ca?=QmNia=@CTo+ku*~zf1%^lvC9kK4{ooPXi*W$527`s@_!-#0CI^nkZKfCp%?*YMlTKyP&WzxE<qTFtJqHg_CoH!IY)eI8C>Dcs4t@4xy(DJu!eK}$pYUT^$(*^ZK?e4uA={>@T`*;`Z<xL0uUX5?(n|uy$ky@dm6{_dxO8r^4{mo^GPaqV*7IveGQ{<)uy{g~L33Rw2+5uZ_)X&IQ5wXSb>mts>_ZghuU+=hmTMa~h<?W9blRLz^cmEDARo@Uswb9=`D)OT+mp*;YpnluY9xmm*0KK@YZgct!+s_8%K)1TVsqr`a9!?a^r4fjHT=0pez2>`$Prn8=1CB%Q+^6@UTAp0pE&ZTJXnKM7{cN$btc(++0DTTS&2w*f`BloaGBi0oLrSF(b~iN_V+-gp=JHK75auvvuCY>=oosLR?!{YjN%L|LDW6`keltQ~)z9{auR6^Ulxz$s?O81(X|s2MT+;Ps?+A&}*U6s!c-8}~_ejaUo<Aa31GDijYldrVf<>z)kj0h7eQ@TOz)rWvLRgvH53d(dp9f#B(fqn&2`2I8V?#&ufO;<gOCC^7@HU!82YK-S5%zA|$}-8e=>2?(@>gjb%4&*uT7rn;2~hzNg$odoqX>$i0_wBxQB|L#=A3Kqw#TJ|%$^yL5z!+vA~g|d3xPc2zp0a`*x0y4or$MmTv|_B+a@td=tK`5%_4c#HwMAouFYY|Aa8|i658bm<WGGKJKX2x$JJgJB1U>4$5~?)HD)7$eev`2L`C59uNopd>&Dza9yfL5r+MXWxpP6>nt)qsq~vn5?hX3uQQ;s)6SZ-KVPspyuAKO2L;)oN+sOeh4S;O3)ZM29{ZhC@O<A&59&W+s!QEO{-!B>7V%%yL*ll42+zW+k0^NFV>oFSNkI3m@yb!)wr}l$oZyrAmtbtcJ_?|)@t8qZpPk@3neTBa%oH{w?FBDP}hH-R1y!)7LpUZ}M?^O?-VAX7MMel{vlg%oK3h({l>76z4I=0>YcC|_B0&^#(;|KqQnq#c!CDThfKZj0G$JKt~_s$s)x#9Qlw(l94vpc_cbTM5611Y*MJ};B<lb>&myb=hxYOukX_^dKUT;_Z+K3(Cy@Dx^>kYVI>c(j?dSfczpywc|WmeeZNR1qqsv^f1}reU|bZS}rHPbp3M;U>9DQ=Q`dHq(pF4gB|jhO&XJxQBF|_G!}1=}exf%?K&n?h$U#rN_0}Ss+^ULtE-8**SnA25OkU6TP+rqKZFwmTslTIF*Q<!_8w*`^~TXKsG)k_qcB+qZ=tCxY<rP-n`uc#%&ZYBfU4`Yy-EF)To|}SCKo-xK(e7d<x+^zEZ_j>sZ!uzC&oG`lc0zcRdi5S_1@k?*!+%fKT~fI{0HfZDwEDw)FLadRV8I#EASm5}z)TQoZ>I**PCSe=z!hftU=qb!!rDxz;7|pPt4Bx3K{(8^4*M2C!1sLde((f*yTmbXD5yLtxWtmluR&KDC`F8M_q>eglG9JVPbWYZ1tM=iqdoi!yL*^nbJGFZIv#M}s!hg_OKEifcCJui_36<H5FDz4wK5X&3MNYQ5*1=w$`@aHyb5pi)oo#lXKjJ1gi+M-}x|x*11sK=I&6MTWoIAYQg;&)22Kvs?3#&1<wCz7`|@bRVc3-(SvTCvA^c8~okvT=^C>TMhbVoH?uC@;@C9h}(kP>aSE5-#z$dQ!$-Zl=Rotc6wefp`gcwcO-qzvCZH`^Dl$8V&0KJbbo5qcG<`~Z7gAoUwZc9Ve6CrNZB4jvRTta=ySuSTkh-CM6F?YQsDe#bv){iwIa6;n{v~&$-*E8ac~a!SbtS=VRSih7V!9vsbkokdhxa}9_sFa6h>!J#qi8~6J{YS>(YjAZphk}TQiY{m4P>O{2pEEp1M~bxwhqUXSrW>3xau`>+sYQgzh|C&hOV4w=7SEst@(e5YI_IpE<iG$(G8)!hJAV?b+vRF<V4q-xnCOR-Qb9@Z9-P>g__;(x^~5$lsWF`8Ve@z2#lBB}6o;HCl3~&OR<zX--v3{ANGj9b9c!;?ON2O)vG~G1lO-$Rg0jLI&ZF4qzAK<Dl&pUS#JMwZcLno>%8ha@)V02QO%!yn@H*`VH{4YrX4uJZX9@ayCB9v{HA|D$V|r>e5OIa&H?swqIZuib726qxa*=J@_+D7n?Joo}OB(Q#H<Jaf9<@&mmSb(HS;B$D-3jS8S=*$*YBG!C7mP{W<p<skp62#SwVl4SK<@O@r9i(_H<y9?illq_zBAEx%$)a{3ucPC#hA98xodn~7+3;XK@-5<06G<{q|PzCRAV(^X@7)Tsu&J_A__x-mM0#t4Q)g)ZcXo(q8#8YN3p{O-b4#2=N^W@oLd6<1QE@5Ns2SxA)1TBup^tYUBZV&(W`ycR0P{7Z#{-t+xpl)T3_sN&{vm*r_o(U+um0SKhk;>kBVuN6S<dF>XuqwcTxvSSgOScC9*^iF(tBY60uRS-Id=5RRV%bE_Ji=}ouuWt*nwkW!l%Aof2BNCH253Q7LV(+R9TNqV$uG%m*_@*QmKaI<8y^+ytg?xbMEq0oK25|OuH(nloJ?te!;pevXLm_rOaQNZj>iiuGv&{@=iqjqPb_x^J7zr`%Sk*Jp-^m|{ZN9n#ILnj;`6K(RyHw!y(Ky{Z{<7na+HD%eb@7hy*Z-K%I)v|u`JhUw6xkT2m_F<k@m0~{T5%UFcl0UCF^!>QNo(r!6w&7VwPcL7ML0F1ytHtVZJjU0;bQz!k}l&cTg>J%jc&cy{XBB?6SeDqqja{Z%uC`@<p%*B!nZ=qtbk%PClt7aNuBEmS9lcOGXM%-y2TR%3SMnfYGMu7STnc5WpX?o%NYw$Jis!3XayP8oOjlz<NRW=rFp-fjV|BNauK|Y%CY0G8%QkSB-5{(Z`WHWxxJ^j3#;1fY3b72`B;w2Yp*ovDw4VpKF{e$lA3%}n#H&~+oj{Cg}NDf4D?HJ8nn}5*(y#vI(0~R-u%YSs)u25UuMu9vPyrCJ{5|BGC^DSY^m-2f?@X-$NWSc&?2(SZM*4q3kL^ZLd0bjaq1nMIUz&^@<=PQ1G4Ek&!SI0*wUntw{~Xnf~LAoxY@dD|FXIVtb;uWOJCQFJ@Zz6RHSLGI<MV=N;%Noc{W_)^;&Qh&(m8O@1DVW<G5hO#wWeBAKYUddabhUZM15yFV5%VhB`;efOru&iugT%aSmtay(otjp&&g*odMr_G+}ebqiAK;7t2Ngrm^sy0966IiIo}ahoblyow;L*DvNrS@^hMnN`z-_<-*Ji+|}xLU9F5q4^Ui~o^&Z|vX2VRXe7+WQL0PNQQBCHqTf}pItGLtQ!6ogywmexPyy7ZUn33Ga`$zu+RUnx)?{X;<h~A8TM}|ZD?|^f{3wJCe-gY=JJVo!x~UwvPJ`GKPE3_iNh3KZ<?}C}_teisZ8cY&n=agiYe5mGmJHYh(3XtZZ0--?Nwc7CVpaTHF1vF(*@sj%D{0SGDO#J^<N!_7`3*bNs>L)$nSOEEuU^86I7;ENt+FBAt$zz1)hc`fzx1)!9R4RU)PzryUNYr+`EKqV8i%RaI&_~uUZmTcZmG16(@W{#>}3Y-9w$|IZB7OOGPo}uEB|gw#J%r}?ykB`MjN_^tLp`GoI*E|ar_|P6(Q~(sON9#l7?HQu#_%%mjyDt9v!mFm&<#sI8&dr@Vd7r{ix$79ZdPrMPu^VI!#{DhR0uMsk_qYA&qpU*_gbyo31<QJyzpO-D&gH-iariQrK%3I!5zb4lFV|!Q{2`FNIbd<A>>J0|kZ@X!c`tS$#}u@SwP!-k&^!_KzDxXO+h0gzXQTd^$=d>e0Mc)b8mqikZPGEEOnVZq|F;?}mJHEY~d;3@>3Y^#W+1U0<@)Liq<VodIG-1q@f1MhooL+P!YzXDM8ZkhCwH3!kBeDRpK5Ju8_{G>;O-&a`ug!M5z*YkbG~t-_;qV0R`9PC=)-?<tN|o#%VI6T?C{<N`@fy<z)`$y8y~ppf;!B=s=SFWJzjJj7%zZQ#ixevY}Ndv24bnWFeb6G3NA=^a*ooZU<1$cWW%FlBrFR9d$N>C{_FAHIe>)hzoi$nn=<4K>3qCcbtn8+zW(dRTWJ87~juhdgtHxq4kk^)4HCo}}%pH2b-Z)n5j)XHI=QEFBKsXLV(R$>+9m!^ag2F6(;nQ)Q0bx14}TvcNKaZM4s6|FowLoxU_|Hq-KvVL!4xIH5_k4nAim{d5|NY=7fUTSyNv%xCE^ZhlWI6PsZf1~{)ZycBR{0TP?;rL0e^%#a4(;glxkeuNaXSq|^k2nL>Xpw!>*t-^fPeeGV0ZQvZ?-K=rSH;)N7kQDXZmIua<s&PXv4O+YVt6sQqiUuuanOU9`!QzBldqQYdABz_eD%JA$<12qo&~d4><CboC@2$#9$o3Hp{j0{yvGFlL_F4;=<2^IrU2RsKo`EBDaeABOD}ccfJQMv=uu?{g$)_|2k?d&0?HPgSR@_Kw^m4CI>E>HlY-4X?$!{$=-w}nYqV6fHjtcl)yxEgQo89)w?xUvp4z;wUE0K9=rCf;DTz|IRmQJU2r?qV@AI-P@F0G{I_&200Mmr;9JI#T0A91*HBA>Fp#p|pu26Ic)9sjk^x9tW>Tj!&-?KjE?a-{kbT0kAW*_$@XLg8}@`ja`p9)g1)a8>iOsCp;pY-twL0xw<jE3Dk7tI&CW?k7K*VPOYGaVVp6Da-MyC|;4ikn~s4+<ZQ!R}yKL@JapBt*@3s0V^0~&LoLeJ!~W$v%Re1c~YDfvAQc|N%IWF%k=1@^&<aepNU!MKeZd+?{B@uf%~-PGf;orw(TDuq>(=F0FnCjn|+-Q>$iut8JRjStd&k;(J_<GRJo*j<R%K@uav;EItyn@t>zB=VXa>4vx}t3D&+jW?L_y>DA?$$nKiH^i3cyJH}2mTJ{=ZHtEs4<!Zz@ms!uO8czY3N`o+M(n^5ZzuuGi$@*u*nCb9k=TN7(#>dHX@F2G_pZartEAe76%u}YRuDv%Cud4<EAXddP3%1E>H(=+u^y305|gtPJQ#whvbJ*s#r)SM_U9rMw3ee~z1x;5$=ksN>6xW;VHb-@vvhD-7>_Rg)rV{K3XIRmQEY06YRsa&}Fq1-8!v2|E7ivjI|J^V62YJ~}Snd9RxIS>vtda0I!a93Gp@dZvl$CW^9>Ha_r?x3g7k((PUr!lczwHKviBb^yAOFirI+W)u(o4U5ClN(@BPUdsF3{Se4QrxYTYvsl}AYf?3sraosVms&71_GXP=>%6#PIx}vqit#d*e>4Z)>T~A#UfTl2OgSEYJ;L{lB3q7yro)Gv)fr;2k(mp+TaFIXYi0TgcVn50V{?$%CF9EYyu0G<%zh|{b0szc~`CHm4tyBK<6=YRyVeTVD=Ozm$&uPen#sLGHWHfv-`V&Gxk6^(8ls@;-%2Qn%-)MPhUEXM;>uaLa<i652@y4RA6{#g^2>aDh!r0XTl~O#9nnu@|Cw3`NkzEU8nbRtUlWRXbl3S1tcHZL!K%V;rEdEg?`Bv$kO_wzL0+hfL2e}GBK@UI_7Sb_23bM;eOlQMq2A-KGZ4*@AJI58DsYmkw1#TLWt$I?%r6gx=Z%=c$Qb&ePzx)AXk6%{TF;b8k#|iOMC7iU?NYZ<>6QTL1S1e2~krVcGNSkZ;3;JQ8lJWaf<!y$V=sz4es6B@DL-mw#;ChyuiioYZml7qYGZwEV0VMn*r$#9HW|v=3u`T)KfR5TMe;1h~?qpTOQ}BFgRd}kIII@N0}upq6FA)w_0|r9f}4c15ffQv`I7SeXOAKpS>H>Yht6Z4~^ulc-#zYP)REVB9ypOpToLG{NaHTopI4f|7?q;5*>#xL3j~J{Y}y|?R)tapolAHpt`YL9EjcOL6}S)kGbA{?r6cL*Do_)r)PJ1twz8}N-gB@*j}lRSM8>IS|1;43{s4L%%>|5K)xjEvt?;#MY212ZJ<dltM+k8`*rI_^=e-|wZYm65v=#E;e-^hDL)E@V}!K4Pj)er-n(}VyFF*@R<hk*1-h-mHm<E1eua_@)DfoBj=h!}cy5lyqhEJ^TPoI3#%AcC%_J~(#{Aoi8!c@CpLG?0=!%64Gl^!ye5N(cZsVYA{1)*js#EZ^jpDw0Bg)<zTZHNAvQrvMaMfy#Po-_2{oN=-yXIc!ZE&lD<=doyDEnAybMic>&CTu{Esgrk{RyF&UsEm44$;tP+?<xQbs32&t`98piuyfW?nnGhZb9@ZFU+~YiZBY@v*v{@zbd%Z0k|{mz6y)Q_#{Qe=@eD?7Xt6!FLG9ir@XB<2lIpSYv$hRWmTl6(z#9p;OgWIM}>upEI@y6*g$8P|EQ8Zxe|%YN_I=|X)QPJwv<ljsb6;*3;`-v^C8v6ny==oNEJ6Zdum&4B^^GSW^UBU6>5GnebIHLOztRe#eeFBpU>^+u)k5hs;^$9&I5Grqd;GNj<Y?qcljgTXvMxtZg%%lYus+Bi)4CVO;nehl;`%Qkx*ctxLzY`SBJJcO1gk}gX}*=(Y$l$z|C7U)kQY#<I?dH`n!pvDxHZtyajG$T8L&_)z}nzy@~nC*XjfVC8K=3za>z4=14T%ro2*a%E=9c-*SAABpg`}s=pyW4@%?RZ4HgQ+xhN2^8*oYXSHsr*_IBh=h@y4vvN(^%AHGAes(07@heknH>eg300~YF2^2k^vLmeC=tfkg?mlmj6tk8G>}JD`w8J^@n<o4lv~Y3wwt)I~c=sAG(`?tm;-`qj=dwH7uKY`#9bM_(KoAe9YDBtzKk7W*-0uT>d0f}L6L=9JW$=7Wd85_gNa^N|F3T$qZ<guDdzH*VXLSs}YE&lo#g6IsW<cB)f0JcvEck>Z%9nAbft82Q<nG7D^JBY@3uMII)-HTKTKc_kM*B{ye0CsvYqd=%*|gd<e|~%I=DSsbk<wmk$a1h)I5v6(c{T0vy9#G7cDj#!cYQCPRFw8oU@+y9BCKSN<}lBAVno)g_f9ShNO2KS)N#MxG6#T!q`9>j8+KT1U;CT)s)+CMVB$~Of;i3u+I{XUd_Cn~KQGw?<51BXP4gS5JUXzcFOC=-m#{rAYUS*1HHXfbZ=3hn7BG%bt#-qYPWXZ&STmtZGQBIQ%;FM$w(|P(y1jo^*WiUiZ5R|jPDd@_S^I=_of6(0{!9Zre@~r+5Cm9Mc`bcLVY}_&C&U5N+irhWjP0{yoBnOxAI7M8JU?@COs-mQ{JfD~huO5W&&wSqC|p1+KLf1U(v_vWYo)4!ghbh1Z;?iNE&$`(?wj(ial7H*T@X_0@p?x6a#&X4roi>jOIEPkHuS3ckCB#{@=#sQv+cntzawOC1Ky%)!79}|r>kB2(SC5Z!>)K}ODnv;x$cKQ>{xKn7qylzUqh69*1Iz?T|(46=XWJ}90y9>3*PHtW0Zd?d)XN4z5^E4!zXzUZSy`f6n6!xlRmGvhU($>^vcuQLz54)DbjXfaa(jaN))S9V(N6ej<;m4KHuNwg>eNyi}8+%-|)1(55uMeR*%2ipHTL=3bIz*CmOZ6_3KXWwYX+h=U}6_ca>Af**vo*{+x20q0N~}zF#z35$dcLW;q|(!}7cmns(zYHrsAgOS5Rwul$;!kECJ!+W_tHeD!dtdVBOtU{#Z8%9}hTf&jUXV%zPS-K$+oH-o(HcBp#a3LDMod~Cny(V~RWRFu+fb`>`1>b8I4(|zV=MN)tM?gt<}yq4{{Io)3Oaqh<hYTont(Pm_wgSo*!OQMxWjoRLY^`o}ho@YDx%ycdBG)>tRl4kDppxYizZG55+9h*x4vj0;Fdwb`wDOOh&_%gmK?{=szGx|I(ee!DqcsycEM&9G*y=fqIJ6=O>*}FvPF<4!_ZtmdOX;X!{_jCu0uJa~UNyG&lk=H=Zlc7fCk730coxkfUb@L6KuejhT`td7GANciX(f}2ehWoDMyt2ty>`w2_*3L(KV`dHY>Y;nHjeVyqT!i0KDf`x&gtyqqQe|H0_nQN9DsA^Fk*1qt@8O4^`!r)`>xm-XNo`vWamO7KLkGE3&conr6sm2!=s?qP+k9>A<XyH3^-R9m+rzRABwocj6u$#*XTKHk(*4+G@HUFE7rl1uT({+R0l2N#3vOi%?5?moam?0kcLo!RID_pAgKsh+lnC++%2%UNEm`=~l&8(feV9YYP8uh^)U=!KNE739So+<tI4=vST3r_zZ`j*8hD{-suV2VL_IcEDiPkTz#tk`C+mV!%Q7KqdhKlg$uXrR}9$YiL6rj)SLU+waZBT#S6L)<^x?Yjnc&h5|!vWX9@GsZo>|2I_*Vl!NgzI9UTwETpiH9KYRfo}7CbDKX!mD9ut#qaYNFjd@25p26*gez;p*y8CBMZ^u9-m2EtquXiMvj+6UZUG`P~RLVs`y^)55HF<^xdFXSfw$*x8ZtGu6bodD0C+)@jZN$x{XtmHsSPfgx?8qd@Z^;%-6G|gidlEMeE~9xk{2zCmPj-Y0HAj;zz*?^1%c)Xcg|m^DW&G&hVEcnvkHa2TMJ|VTyDYYQ<0A9=Bu3J%z~ko-~G<dD^TrrvlbfkNgI9#OtN2>Jq}c&BYUh5}kffhM<^7;KwCTqv&P>#O`p6j-9zC7mvRM7H~XkF08qbtSk7HjRqT+p)HJbY_rBCZ3+QHcwj5z8<?v@C>fu^6_55#PI~CocbQe5RG)pS*D>=u%Z|a(=vr1qw~bYA5fA*TfZKT|fv?AIE+EyiNfUXT;SrBRnc2l!Y+{VIB@d{cDBfiYlN`EWii*P!_&Y`)QsT){vwTFr=_LC&n`SJfK1)o9BOke&0b>VvuPLq$wQTexGOq1}W@2!)>^}J@(|z8`k2#xbNu437rCbS)e*IykXCWW|7?He}hv{#QLM2oeE50<=W)d)^If^DfPk`n?y?d9-r30xpTWZw)Tp{U6ESaDy$T|qVpvSe9BCOBcnq7su7d6Uu`3+yKZX>@-DtuB2eVtQ~U*7V1WNBJ$Ev<4<dCvGxG8+~RL=NG)hu8P8y{0dnTBC=V>2+EkbW4w>GF&MpD-)$sZ@G{A{2b*6h2~>ED%Uk-*`RL7<ceUy<n-Lm`5{rfepC5k(^LV&tsCY!=&^qHpho5TW!ochJq`9Vxd3NOW75j^m++kZRu7jpKOcB4IPO~mzl!f?-MU9V4^Qhfm!S@RU`?`EQ2l9R3a22)0tdNPllk12Qfh#0$Q1N^{RWm-w@M@*PPsoH&7=FWo$enep<hJNGJUtcs!Cs(qgH^YKve1U=DQ7!?;4BN%o}Y=d6K*Q`DX#-=C9~V+rmQ>ipABVI(jj7V}>jT^)mO$?*=;8qRFErl+q&F8crr=x0YDfbFpj8+t(TC$GQ~alWq*e1((M=T6KS>@_BCl9?gz{17tLn^N)2_$@Dx-D75dApjP`FB6a7l%rGJGOB-w5GNnflOTg4A^L&qYYU?))ACL8Bt<Uru%L})(QISZhdV_Zj$}IIW7vCqtHK9GH?IbQonD?lHBk7DBZf3SE>{=b%P|!zrK;P+pc|U)Rq+6*Qk?Z|F8U^>Ospdwn@vRQ!#P-p^Cdw=^4qO7g$T63eH^*1ME8{AYr`a)9-)3A`x74dP2Gy9UO&&^~SW5XGQq$g);cMRU{FYM*-yfeY+j0s-@STX(y{v_Yw|j@wX3+tZDMWPlf_UqLRTP_q00s1Ua|<+0-BEUtqYIVILNNQrSw5!fqSjw(=R%9p!tVYN6)zoD;cq7bpY-m%*-{e?Ig<Iyw~_&H_l^?0ysUe-^+_ko9<2VTo!6KI74Nz_Y?w<9*-xRc7*3A1Vud?2?-SL;k?l!$HEz)R1oQ6-ai@;?d$e7f^H=4U7stUT%tp3RC8xl`7*4{rn}v^V0k!mFGvj_=T#sghA2$<2VG%Xe>fq+ooOHYodG@tjK8!@}ad`VvoV<utmGOZSEl58ERna@L^|7YPvYwVx=~>(#Fr=GPL0uauT``;!a?pB|fBdK4sz36A5*+yEg}o<r>20P52WCn0qAGw5?BBAw)k+&VY3wiU58y#PAWIy+)m>5vOUDtqLTWo*92`IQV&3KRLwl{&<$apfc=JtlOBcC2?oi)Ase&!w)xZM0WAlLT)d6Y4TdBr&s*}nPf0yf<ABQs})%SIs%{QNR)1o->gl0o=5uM$}&J63-*|dm!(DLj04tmoS@pxgit8mqS<58>Hv=zSZQM1#ig&^AL3lo-aNhe-=;9F4c3qwAB;K*{ei4`HI>7In=!wZTn+e)ozWPM@lI6SaAir2AzSmwc!EL`Z}^L`(L78a>Rvr+i&*juAYy6l^Z_T$AX=SBG_6=AyPABJ$UoWF5vP^63;Wqm_SEeArRF(PI=qo(z*kFxx#_T~%G<}?EJBqF0&AHRVH%V8zr%%l>yvF4le@Ou+ebngyUF43E8e=j^J7o+-fkYHV$bq7grhPj9bjJKsc|LKdoG;Y4SdXGT2*9Ep4OJ5Mv-jW5gD|6s_>t>YWYvG6AacwPwHiymt&hT*7>Ym<?b{hmjXa9U1zQ-op3@XWului@&>_0Jc8$7`(GW!k0d|GnZB7^%@GeP<RSpSqA^KNDq1}yne4d^QJ<B7p@<5_(j0xzeB9XTL#{*;yLsZIgcF)ou@ICqo+-0qHXt2;)J=lRp!17o{?Zu>>xt7Eu8u<OM|b@6I8#JYvun|-k5P&xvZRJ?lg>iDTtAgKD(+!k`Lcg?)(tNut#ao0DmyQl{EOyMrhOQdWT7Q01mle?43nYVJDHbB;~7TnYPy>Vm_bdqiB;shi|s)o>b@Xbc8F*cYLPrnxh+f3Wg5&hJX3o*l>Z6+m!GSpUw?5J$b_s8S5lxd9xh2(%-&s|&xLd|pE+;26O=^58fls;C#+8phkg#T0&+Ff~Od8F!?+F3KfVDP?9vqnG)`(W_H-JlVuVBG=Lw}4tYxZqxL%&xbEyjwc{JC8ap*{@BGd0sjU8)-`<T2SNSWEWQ2X?c9z9sZ;x|JQ#VP3v#k(KN!N310Nn5B1~U|7JcfrV*P*?dIrcqY3?=e~sxDCXr$OjriA<cmMcjoAV_yZ#VGY_v8KJ&~E(c=&!&2;e~GS?GXBh8T|Wq{bsoCUl0WSt3%iNUZ(ete*=O4hJgOzn9rkYo3|K*{#GFV_jvw2oPYjBx_dMKCL}@Zp3PAdg%S9dc|MrOqj~&}_n#1dJG-~D9vjYI|3;3?IEjKkPyE+^uJ2oIgyHCChF^*N&nxr({<r^e&EKD_?*F^df6Mj$Y4V?I{Gcz+a{cpo{+Ff&`q|X9zy8nP{?cxt>udkIYEk?5&4+*e{pVNBa{WI~;NOW={`FMvulK+4|35p6jK=?OS3l4Evm^6q8sI<Q=g%YH{}9IC|6hQAe*0&Gzo1`%EZ5Ng>f=9>`TPHc_ut$9dj`<IUcu_e3Xg=vZ28}BQvB<nH?d~Zsu>4VHxAW~-f35jR;`8UC|W%n^oDup==FncHCsphU^J{|9kUu}-LiiEgEi2<{&&~^I}QIYqY}A$q?%ZH;KS>@ap^}<O%;eoa}S=>fw>GgYcIVPR*u#i_I6Lp`E~SO&wT)`{PpiHHv75RUSKmV^M71(Di7}dyA|biw#Nj!*KkYreee#Gs>*q%mq~vHC7`&M{THgc(P2O7Hul+K`^@<@nE)M{_5mvcDK0&}XqCSp5Irg`s6S9a<D6WylRha}gNNhw*4LJrpY1ZKN}L1mqxpLnRjUl=18t{lVN!N&7T5G0a7k$MnoXE_`0AZM8q@v+3rOW=nF$n^)*#BEsms^3{8;UL{y^>1!L_<rD1Nnf-n1-hkWL2IV#06YKKt>y`Kpg7Ww<$8-PQZ$HNjASzvxZFGtxiY3+=Qd%G-ol?9cZ8h4oTFoT9|#biZ%%$5(C|+G6KtF+Y?I%DztTWysC7fN+iDQF`^gN0YULv$b9J&H{SBPXKj)0);qV?e%KOsg|M5*`Cz%V##6GEo}XGEwXs2thFkB6dQZ}weX1b^AA+BkQklAa)Gc!+^3a1WBfr2M3Q==_Ac8VC?*uLwP*Lt&tq$^GeGKX9wPAC326OU#{_<0BHLuO;+5y)lp^8M-|Mr|gYTX@$t0bM>s6%wfr3lQc^S;rGAsi#RQd7MA<!4*>iy8~%vBAfhW%<wz=1F#+<=_#-r2(EcI$DgIqjM7y6$A1E1%)DnjII2Vg#O;_gh$JwUmQY#~)g_3rT;VFuRz}I`FirOhKZ0wUDI}4R(W_e(S)auatSve0n3S&0nRKp!wADwdc)J!iU5y^j5l&4yQL~=(>Uigj2WcO~C<nWK96Qopgo0_G!-_c#Vjs@Y~3KN|JVE!oCwKH4cBDE<s`67WqF=K#P6sBz?5*Y-;P&kDsD-V)*nWPiRco!I{HBzL(BS11qwHp^WF_O0(SJy056FnRY@dQQ<pkV-oj|E2~%RIwU|o+N!S)pGoP}s-kN|x7V}RLb2O3=n#m7YTsDV_^w=~;LS#DcWD9e-m`7NmD9;Klx4x{B{Q17c-8=E-jHHuk2_HY34fq)o2iM?)c|&alpTWzcoq<I<DRh>XgpC(E)eHI{Uo=mkg_a>!Rn9=B$<6_7kXch>6g*pa`)$J@g4d8-C#_ieOuaWjis}AK46}`YK`A7vfv?{99fWH(>m&l&pU`xruZ(On7PDIqv^K?yIDYpi9Y)5&(UaoSvRkBw{t(;U{Gn2$&No2(WJAy!bDfT8f9<e0Pu<ycLtL8gaWje)4SM9r`Yj+3Zxi*ZRT6})oCN!_h@=R%_%wC+>Jj_r~WzSb!Sr|M>f(NhBJGLjln7j_r8O9=u!h^=V`y_JvAi5I|c4GCDq>PGtK2<ufC82*~Ab+c2!-YUx(hY&h)_%ql)${U|^xUjDmQ^y*7Bz@#&6L(%OL1&rkH3X|}uMrmIHW_nlYjYiZmZt{d96*?*PJ3(`!Q^9n}#lZsy)_n}x1E6c6Uqi}wG#6lgJlJ}Q+O3E|^PJfs)=_GR1W+kot9lnr`_Q0+PGO5@d=jUGGnlFl{hrt1Lx76@jm=;tafUoS$kUoDK6}TtZ$)zQ5n6{T}J1RmmZVC|Q_<9=?o8oe@JPzF*^#W&|S(J}<;f-f{yS749KQiBrca*cmAVq(B%hL1?@U_ywZ}gcvR^_1Hrd9W66#n=F^{g5UOhOGzT+>)dIYwL1GF-5RVQ2WRDFwZ~yc&jd7)~={b!v1PuU$~KFQ@Q?E!{-54<VK%R%fa^&ot`2oABy&Y(1)26W(3v8Y)j$jpf%&-L_P!0-fF@)NwO54{Zx~z=@cY*R<w|X}TdE#(q8C6GzxS(WgzjwK#=OrngK3$iXunYJ!;q*9MEUf9-r_)f(DT4l(_Ce0bJt>>cvP{TU*qS@O?Xb;E6a2HEa*&@4=RnDedn$hVjz?E7r(5b8+WT$pwtRu5A=IK(A&(Y~jkn$8uMdSVTvN~HOO^!`9`3-9(}xGKL<p*<EI7xDB7d*Keev!8Kroq~C(pm)a9hb%OqG+B#YPXP{ub^=C9sd(Sj*3{qubh@DvOpc3|pdW5~(Ccn|2S=-1p#)h4_z%=|H5otLwC0qDm#BqpN7lieaXaMLTKCKJ;VDI}MDPw(VcyUD7k)1~_Mo=S;Fw;;2Z}~*Lkp{hWMrb&INb+;sea`Jx|ip;Y3G#bdhO+96ay*6OSSRHe1XadeG|j3_ga~U!>x$PBQvN7sdLjO*K_R81B-3LHXE+=YI{5`(Dr%Pj?ar$$ss^^_3SLhgK!hFUwQHMJc6r099`=&dg=SS<~Vw>$z?$V)9kq?_(Tbcn~L|*k!5<iXj)boDl^(&$Wk2+u#L#}-eIwSpTzLlht;cIQM2zfkhSVI&p^`{9DqT-tip~xW^Cvtm4mzq)!7qffk8z_KX&)qMA6|NEgbDm5YSqNbG?7NUb5P%RiAV(>C^3(rjkz&_43?4(x=2l82-Uvh(|1>wh7l}-Rq$xp_))!Hx#Sbyo~U<QLU%Z8ywd1tNleWJG;*~%e%%j^v(|Qtj~{LNk2%{7m8>~<IQk~4<&>>K{li~LhRCJb>DH%DKvPfXY<Lp*IpRFQ%1In2%Xl@CIxmfyrdyt0J+|;hLp~0!2F8tFnwk4NfGIMMjvv>uj%YT7!!lwPie#``lvTrX7}RDVRym+3W~FGI~=Gtt~}?24NA;L)NX?~y5DV8*9rCTZ(+x!0OISlv>OHRhcsmyGbkLa-UWQu-@VtT-rkjxX{T#Vr=L;tDEutq(Hd~k>AC%An?K)<xoQECB%6c%cwZuDN_ix$*{~{Wbl%+}chU9UKwlTt4UPYS3Kz2BRh<+k13?XU-FG^`S%s?}w~ckkzOKn|7+%qC=P8iYwKqiwSYNTrXvBOLfEevr=RAL{#dCeA<x>6d<B&aq;f6KsJ{j{T%_!j><o<zLur=$@;t-om3dd|O%=~7j3(qIg0n63^V`CDtnB83?^*Ht*iktdZ@>Kiw$eyno{bP(B&KMeM|AqZHH-Vu{brxYYh#3*b1n#Y=o>}!|tp{}Q!+cZ`el07k3Oor8=m{Gdq_%UbVDAXp{1tU#@LFmUdV@3Q8^l!Pt2uskf@^+kzlUKSh1*K<s<YJO+f{Gas}7{c?tb;Q>Dj7Q^>>J_&-^4)Y^E4qE(Xy2AnWNnx_z7A=ViKS*Qnj;TsyVNRWNFu?l<I+prMmt+ZEe|-yc*k6pl6c8rVJ#uV6Q)^6A#v2}EFOr|XNXq2yZp6v`|I`fM}1g_yr~-nZj)HZ3+u@efpqYBCMX-fQ)nF%-J@YrgJxaFpvX6RXLlkI&|IK=SUPu&vE@rK0GcrfNu<XKp}9Q>Kw0AF~oV9&#}%ftNuxXZIlKmqup@Kv9OP(0*Uq|AE?-ui{!)&e(ZGM>@E>3`b3qdkpZc<lUvl*lIpCHhJzo7(_KD$YS)v)@$otkD6`Mjh5ZsZiA#|Rb#-(gSnlZV{b6Phd+qc2a#rd@=z^J|3Kx)2w86YuRH-lxqVDkxjm$4SNITedA5H{u*WPO*84}zmsSM95O=x^q-%(XN65K$Zb=VdR>zz<`dOUPW!Twx!P`<Ny@>-ni=PXCh!^;6K=*B0>j=H?H_MDAWLgx94K~QBt;JHW`hyA)icUZ2fn@B@@s$VX#^b@PY%Rri9$XIx60q&IO;kbZtIP2wI(vv-tNT_$cbU4RplfK^DK`wvLIJZ2b6+Q#IdS^>m;)LYbeof&(o(1)GD#Z2ATrsX)l0eQ*%&^N+hVXQ;!Qv|evEgG;;Ogj)wv$o1N+&bPX3_hTFG@!pWrL56>PNDCR>!V>61ge@KPXxVXkWMtiC0?uH1D$jrh_1sO#Efm^GF0Ni1R#MX#zE@3!BE0;~X`cLZ<8B0f71{(aE+Wpej6OVw$+8{tNoN`5Tjuz5G5DbB%nYxPuDz>NX%nzULs*7*h+0O8f*jyY}!+n&WQ4b7?6581UMr`^KdHPo2mhT8>MZ(t*3#q6Oh93TEb^?ANEG)L!}Pq;uWpnupwW6Th@liNeNQSpN1Rd<d!Q!<WPLz+a^;~Lgt1}A+Tg_quEv$)6EDq@S1LFFZ()Dm?V<Vmj++gN>PL-rH({siF3{Qlzuyyuk#@(w2f7$WiXY@5ul4Se6{n>Ip~NGr5&9SyfeV!D}c?nOM@lXR!_fsnPW#QLYr-BrRZH{BJ=1@xf!s;NtBDNwt@IZlDx+sxi&3|X;lgW+wXKb>!q3EmMGMCZ%<YnRs!Yl3G`d<`a&teqS9roLgUH7ob|b|F}7lGlDc*Ou{Zz1J#A3$46bNA$h?;EVGVI>M!DT_t`=;dC%AZYFc`><QMfyBJC>x6!(n%K~R%I@<lw;VUz^&u-B9GvPt}z1BV{qg5V9H+YALynNx<TpZ2(;sZq$v^w&`J+=z(w{o!5Aa~PzzVmO$0$khk@oh@DU~BO>bP7M5*<6D)()JYgR9yEancKgs>$wz)qkH7`3?DwSJF!wi-IYVU+0TurbDPJscs?L&6UCDQ?b-Y7r_=uRskBjNzNs(YmqLOl>!YK|+3hnwGrIJMb~CG;yX2{tAkw?~Ad|)Q_-$8zBeI#3XWx~&*|=n%mz~lH^kx!n?2;#W&EN6Vy$sS;i=x<sJpC*+mR{TqF8LVRFt2}lK+LW?-M%E9*@3pCwV~eg=kT@te9_h&i{E7a2g)n-`|>ZY<2?p)&ARP~{X!UhT48}K>HQ}vaXqGgO~?XPR|Btf1{auBTodA;ceuWKujd4t4m-f({Olmfb-?s{vt>QLh&eGWoRcoxgxd0=fA4SC=+}{roBF9Wx%ZxV#@2fC{otLhUqaX_y}A!;-<tK%u~_SP!u||%QCWU9uZ@E^Y5(Zp9=r-<k~EB+dNq%q#l57yi60W)Zy=|wzAdDqKun40cy;-DD}7$VMT%<@K?8<+^vsziSx2>)-RK2&*3BhxSzNZXY%JU$yHI!gQma)*1WnBO*$!2VJ`LB``$fzqKEyZDBfGrIAvknBT86qHbFXUeFH<bB)}WLJi}NEkV5V1n%tuo)0m>9v8)C)460GVEyKzVhC^d7B&{uhSEwE<69wwxCR(>V~KcMldwJ(oH64C6&UgD(pe0(hdeRYbs^Y&}-2a4@h$vy=Q<YPU&REOgdx|0{+9VOUpk)8}E+f=&9Aq19gy*}G1tKE$ZK~dN_ctE{<)JE&0F4y$aSZ)n|9T^ABym_!@odNM$-Kup~DBfqH`h0b-!N?mFmhDc^x|DG>8_UBV|De&z7hd;@dPcc&0ge^vQ!SM-0((^J;v{ww%#WFd$oAtez!q^GitHTUEp9i@oe!oH&DT8IJU5z7v=1msReNeJg9+dqJ?3%0r>4zqGb`WgP^}={-!@lV6^7)uyFlsTiTNBT?xF!rtC8pjz~((bOs);EsrpLDx;Xw6+OyNlzMtp)oDXC}Oz9s5%4RpNID~&(zOg)dVM$9y{S}It9C8_A3GB5o{AnML#`n!al2i}k{Sfdc^*!}VH(m{09O9!b3ja7;c`Mw#JSBY*5P4+)wra;s7`#b(+NcZFlqzSu2~WG_c5q_fMMRxvT%eU-m&0kVoaqSiR!Nf`LU5qe&e~CM*VMB2<1YV!Dvt+kd&uzR=NFt%=9A@Zx|<Hy>LQChFu<{^(=R1SOpYB-QM~iDIYUqzJ~HpJ)5}A^Zucf+&=c`FxQP2`2YNKuOZc#ABFC<^Y#eawoPwLUJAvd;vsJLZYC?GJqFcW)Jvrsiq+$C#G~3(gsbhvav)=oxupf?98vumn@pNHTAdbVQ+a!_4rb{h<-+rfa&p>57frdVV1BGJNEPQ%q7qFWKyZ$Wk>ftK&dY!v<`1#zn!^Y!T1=w$6sl$uUdfUC9P<soW)bk36>JGPG8#NnD9+$^0pzHf$wKTh&>bbPDpqDkn#uh#6Lcrpo&w?&Y&J7WMG>GHj-Iwc5w~ql)ds>^Pr;pUcf6Xj5r4A+?`rAoba;~6Ju8B;TPMdvoz7yYl!K3QbMzq$?n$mu)LD7R|%VZndTw%#)ogsE-CrIr;z?$`*FE&<tGrUQtW7gcFV+oqYG~&h|sPR5`)z7Vv1K@|tcTZRWJJa<hRU#<vRq=>l65t1HHTH{FC_SA!+%Sd6UR9~(j*YQreKId5{T_Ts@8|)u5vWtZ^_pCjcCA;`#vZ+7Y`nhV-+p1ze+FI$Ggt_0X)<zj7G)PO)AK}d{R#bK=eyhNrR1{8jr-T`Wu)mYe2eN*bGBy4o6&Ql+b%51Tw=GN<9y`vXnIT6Z8Sv&>4C3l-`LUm$F!x5ywl`rhv(YQ3iOWvru<$CWEZ_13*idKPzUvr(ok&?%ynM$q8t7Am^prVZ1(9?EV9@cJRr4p_tv#q{qz)L_ch&nuIm2%uJz~1?hllB>S3zOfDfN^Pxx3h&~n^mbp2f~``LQ(-nfRj(+@|(dHdc$YBlu6hlPF!9d<jl1`81x(Cq3pk*nw%e>~@)F>T|Rgoy$Kaa*T)FZG<{57e!ZhHwTYJ3gWyqA!CCIJTI)CXego<AQvTZmc!zR1e8<yXt56>x3Ldp{IBi2mJL1z(5(-CX6>qqK8T*T27NoW+L@6R!)ax;n2m#6m58y59t2J{L+HZJzK4!Ji7k#YT$nAU1&3D^UB=pAYxmEU{h`OuT}IB!l-)P`6a5|Wv{QUJR@IE>>c%KcjW5bsH4>f{NQ-|te20ba^o59jG-%Wr?me-Whe{5Xp-3rY<swc=49cxF!kD(N0+)X$_96}>+^>KL6Tec@S+=jWG7B{#ZsF)Iop@1wbGt9`1F}l;j4O}EJyYP@&f&=jO${;lk;AFLca0cCiDQe*if(qtx>eML9n8yRR4K>Xs?@jaf`%v{t!;3)W|--^m;Co>MqyiM0QNyt7xa6K+n!|)Gz<AoUse%+kugEu2_tmT8jy5?vr-SV_OY;ryWgHZgcivvUHzi1iYWw*-EZmiyksNw1;1EnqV)ruuc;6<PK=Dvxj=#XSrWA{OKj~UXRx!4fn|6so(OJ!qYjrrHHamU)aVJ`i^!Xwx+9BN$+6c0gSF#Ep8kbVQgUR$Gq0g$Wuj(-cHx#!CMdp(q(fbOGv5b?Uv=uZ)BUg4|ff#=!ISB|6%HDyUq2!uU|_;HB`eGA*F%b6;W~%N)IX|q!N|x@Y?_9bB*sgzw7zQK6ZPpHRoJ&ZC>BPhDY6)m9i8@X5i~na@O>Uy50uKxbW+xj>_ixehIv_#Vt8Kf>B=uk4V~FT-!6_s1ym_>fb5IQQe7<(DGmHXpvP5Z>`4HXFj|oaYMTnIa9d(S?!4=y}^yC$|BI*5Am+#3Zj+{XXuT}6*C#4_<KMi7Clz-e*N<I#Vuya&=Fj2lZ@Xf6KYkZvu>ueJ+7|Rh|Ln|K*<j#Qt(h|^-7cL)d=zOUU`5UqL%AML_F^3E>MD{1Z+;tcyc%(7wRP$zYCz2^t}k}jdw!#eO8EF3;T<!uvT3*_rkkea1<YghEOIlVb5F`D@Dq_m}Fgf|7h5`_C2BI^i`T%&cOh8(Ay7r7}Ru}j6Q&<r_ytWHKh)qNf|JJUfKFOr?ilcHVx+7`fsUM3A*}>Fpe2~VBMYxn5sW?L7RxnVXd^^kn_WeubSgZQ|A5(3Kq2Dx#6$6GMX0r;ot``SC9ka%w{VG`Ciy9+}M95|9mCkKCQQZz{t6}M>nTa5W15VyPhw@({S)X!vsAJHqVkQF02}f*4r<nCijOUfS&!TjOsl+l^(>zl+e8I+QWTK-Y8_M0nqqRXx;IOCmrmx*<T^yHty<cthLC4=b-lWTZGqs?y2q`2OV)vk5++)R56M1)JA=KF26#R>v#*F_-)ZxIkqyu@_1z!qxsrO2K`l`pD11oS~%l6(O?hHCSl*K^#@Ib>j|ewdc3<*7Y^BFuWL;f`NdE2JT2Ga+dLtlDW95~rRytlfkDFW-sP`Zul10j03tHQZls)n(?=;=t>5gzdiJ-$RSVY%KfKS)#qTrXpvR7Y_GIJLXQiFFfT8X)%J18glcWnC0=16w#j`xz;^<I4%#JDmfDd=8+(#mNeMS3-aCM|!M$%J(Da`6BoYr}z#6DVG`Sf?>G$6yJ@OoDZcDTBt;^!TuW?0=#=25DbCz}S3rkyz75`%s5+P<R7JXd-pz=WO-F(>u+rONbu+*#4qaRj*|@_9aaL-E*2xh?qDiGKeO_<C!dqtMdB!1h>^aOiY+!N^3r&{i9?eEA1)d5u0beT4qR?KWqR*<eH`wsK9i0mf7l7ejYIG0{bGgYQ6Q)?b_~?7bjjg!wZH)Sid7ffe&%K1`?m?6I&LvtY~-uiZ>xxMsKdWmm7kvSWd9bKMfDZEd}!XBfCV&I~xe`O5qVn8e_A|85sg>)5V&HJ9_3uM9?`RATHwY9qf|<Eyrx=HWT|3Gw>dqV!-c^VfXY2>ao4Q$d61BX@#*&+$(8>b~@P<dqd>WWr!rqTM&^z8Q=Es}1YbZLiS7r%XL&x=Or|q^_z_eO4tzt`%f<*jIZy*k_CocsnO=_g)-t=;reH_6gw+!UM^yLpA8$E*Bl%WU>H4h3?*~!60g8=$dP&k+$A7JHdOLVeTEm6+#~(H?1@jq$jJM9Dm!~-3Y#O@Ydq^3e|um0CRp>D>st5O{LVN{>wz0jdN@F9xse0frveSQ~Ak&TohT{5nL7b4kFj+`m}qatAXDs0tNIQpSmW@ra9_=SK}V9((0^omOwJueFbMc<5!N{iWOd7T_erIU)=ZN{pRDM)N#s^IGwtlO<(Y2NhZQuIlg++PRuN5%YWpTCyr@vRJ(9bUsWxAB<=c5z1tP0w_Eq;d{jUgtWE6rVaX+R6u_|w_*bhROXI^|*I~bIpiCFI^fq(50yheP?STd+bfRo8#>(KYyR<yicM|u>+6lY|M>3GLX646vke}|a;^%1gg|!F~%O7ybJQx^vn&vViHHVMJX!Cw9k2VYRxj_e;-m@zN_JNkzYZFTsd3ErUfcR|PI-jXux__n<p=wsqVRP&A%fa?r170uzZ1Kgu?U+=s>Kwn{lSvyUK>43Rf7zQxA1+qztmVnyTI~ShZ)IK|?3N#jTwF@FWg3EHzV}W8++WIrGkE!ZpRXDHT=whyvbToI07%?+$0K<bgln9_%k^L9eT<ssmzDC@hnd2QYnF`;pvryc{{A{2Oub0VpB-27LWMkC)Ht7iZ?hxJV$g9HTu*wRIg{+=dYhQ<@9`7uh&-;a!9_0aZy8CDk@$X5zr9q<^1H2@QJw)>x1r0=`FwDnwabJ?^VjU$J%J)4D%&)w_`M9f*06=0lR8yzJYc&%XNR|-M3z#4rlRP7wRN<?dQm;ztYNe`l!K!Wzi_j!tdow@Q>h=AvjI9o`(@w8BkO$vvhnE=#&b%7#s#kl&Ky*y$90_)ezy3)Ty0}~s*y-aF#I#@`1fY=od35}$z|@v!DJBbK#T3eds6N0k2;*64>P`}Jv?>QQZ`~`cR^UpJs>5!+2uA8u)@~zdp9VALNr>jT-0ZC&jLq3n?ACa+o9j%N)!`r$`oH*e#O7{GFiMIpB&FN_cu3~l>hp4bMIRR$H;=!Yoy$@a#2!47)mg>UXS<h02*l3Lye3{3~+k8WxBGnEqpV~+O{kX4`hK?2LpY3K0K+zVdqf6buG(EowD?Yl(sdiTF+Zh{;Q~m$|-n`S|~-J?G5#wQui6_L4#YYc@D*uwy3ZPu@W7KGTFSK@69LZq<d%8Y+Wam-c80jGzG^>xg_0&C!*>0>)nZ8zq0G&Wp*@AxIGPji?wU6w;8~}EAMe#cKGuLfxmCgT2v&QxUv@?KiQRWC;J#nZWBQ4?$CXnoBM$8mDXi<M+EOzui#vvcdXRUOdf>5!Bzk8>Pc6_E?f8V2<Otlq~R>t&TO*TYE56|p|FGDTqNt+s7oJiWmcRvG!5#a*cK_AnK?(WvRB*#pzaF_eQV@*3+QQ4WtW4>>x^_{81|3d|CSnD@m*&s0M_*SypaB50fCl1e)MH*hl5IU(COVv+v*LXXWf}#?wcK$co_>5yw)bCTjuTE%X>Mo$t*w?v$A-cmVW?gb%nI}+<j{3b24Ap(68K5irlBc8rrw*bQg{%?xa$MY#oqn1?qCK7wPVttw7_N{sAa;GCZ|`qeNN!q)aGly_P6C3mzw6KOe1c(TtKN((q~Z(cewO!N=^i6D(H|NL%*fSM9l)BX}me_F!zgwNm+ow(=V~qI#$2Q6$d2cqglw_-%fUm5CX>en03Vt}ng?uI1yN&JzAUM>?w{7Mj=X3(z8KX>O+H-4itYUY~#Wf1hI8Ca81KK4wxFn=#?=U<ewid%#A%y>lry7@_#`m-PL;IHy%Q+N#SUe>@{xxfX<p_W2j^dTo1L@#H3(X0CiL=IwGI%aDm@<xj4LmiAkT!2RpJnIuYg=K=}be@(e(AQl<4s@lh)j2D~Pd)neU%UQ#Q>+w_C%41}*^H~`ZXYGqLfhN0ASP4JGd7{7l`*8VPR{t72((}^IKl`a#Z+N3#a%y`%(V?Afq+N%d#_fLO3M<b%a^Y2XpUWO7wd@Yhj~VG8)36NJ`_wWyKX7Un&`4S5&Y%^l-_(liXNFbTmA<-Do8JTRgFJ&7)Sbf03N8g-yOYnylTkmxXYcGoZFV$?8!@{UMjcjv-KGyW9A|?=`jl3KXbs?EcP*blVcd(aoy+{#^@T5q2gAP7FuO00j^*Gb{&L>#sqo4@b-%!S`gT`l<6E#Yw>RMpVIft4PLm~V0`>Y9uJS8yqA0)HLG;P54DS~cPA51S;YVkLz=r#Sb6B!|7mN3v<zOOsJ0ZBq;d}IW=stLMi9*m4T4fAqksS5*krtrwp?uIYA^`P-rc3wcvt}ia&E+f9{#$AscR}3HjYW3wxYeMOpMUTJyO=o*{~A?C_;sD8JG(%Ca{YZ3_xJ@XdIOa{cakyqLk@WUW^K?$3K)~yKx<6zX=STc2R#NUur{pDqTO=-Th!=M`Qig3hW|Fak_!eZmh>39Ea0euTW#Z*A3(sVT_@b#86{V1`}v8tXNcDHcklVhU`OFpQ~JVE*+&HSQk9EhOPp~_tp09Ws@A1$yQdyFh++FXTi?!KGVZI`Y>u$fk$}*DzxkaEwEHdO<K6u@d_+Z%M8id4vD*(laqbcwi=!!5ZV2+KX9>&2qvwWd^4H|lbHSOsayx1d$8Yw)v#Ah|rxVxTT0e*VDo{$a%`Yi!rgQQ(J(1n@=shdpS4&*Es`btp5ZOqvh6L1Y{#6{3>#B!xYeH7V-uPClj!fSErKsO;(b<a*mC?F~hS`y`j^NxwlFW?EQ)_>^g)XYe-{BNo+{SnZ+r2sxhXa%PWi3~CYtr4ml0jf>X9&;8w~hWu!fyk#rT)hK;PuWXCTiht#U8hP6yR;nfvy0497m;OtnAoZYxQ0$mpV#nBX9hzue^5s$Su2mbG@_(yjzb1Q&SElIkcTFJ7@H^STvw%dKjJc&U9-g<Yt#K@MGnl)n6k>`}#Sis&EO-aj*4ia<kiQ#8E+>zR2sQw*{7KGMaouAi78l-R+RV<g<HL@vhPI>zD4cxU8Y<?|m~T&+TCQ8h=)YZ?g|MSLQf;7a#V_;%s-#EG<4-cgwQ1U9I=}*uEYcU9A>LYyBA3e`0aeeX#WT)BUcL#gRi9uD81MXA?U*UT!fMl*`nwiBn=~KM-YY3)a~mhg=JJ8@Kb2vzx5b7L-!7FC2_353b!oFMg!`5bu(6P}^wd1imp1X7`D$&wF4@{#tZGqxg5vxtW_etFNDlT+B*+|54jzyYBF&0PK=|iggHl0w&0T0)Z`KyL25aqL}6No%ssfyR`Gw$!y-ni^bD?o$Hfs7GVKv<2{J2VcsuBTAlsXbw8+#ztuyhS%6fV7rlDsveSBdED5{(r>g8#<ubhYK#1LU3mlOD{?F?df;YKB?yK)vW92RBbgS&dfrXS4*o9L`C|JvWW7;tt(;t=eiz|9~h<x$7#=QM)05bo+=THAD9S`>TQ1dSPK#8R_Hl>vM{qkj$!uz$8H9w~!lLM&x!J3=FLE5(DV61k`)tjy7`;6Q>C3u;#4_$5gmgFmP{m``-snr5YLrZb4>=WCeRs7-3N~7|}Z~^3dZ0=6)4cP5jV`^FA%DrE%h~AERt)dLUcJdmfW9fF1ul;Fw?LR7uXag>c2C(p(?`=7(i}C%~&GTlsdd@!}=FO>ptu}B}ns;RaEg?Y@s&zHKrNW3w@g=|eB%~f3zJ5I=tG9KR_%X%_>n&y%Z-;Z%K}kdWRhvH69=uvvUPcE8uk~6)Y8ssGS{vNB&94`UBL((~Gww1GZguK<X7~OAvY3B{8QzHCmxrVJGi82Ti4%TkHo~gCSA~AV+j>%^w!G0T=N-*IG;P1(-<K*ce4lX|j6^$Ncpf#F+tKedLlNU{q@hJ(zZu+K==KU^uJV)Hr6XT0m|5e9Qms2-oL1bfR8x+y(-7lM*H*?YPQxx_*0aZI9$!(Zj1j-^i@{qMY(Nw|e=nBkAlE()avdO6wLfc}ouZg*=oPu4Y5dU}#sOpxyKDXnxjSTU#f=}LN|minHq~$tUEin&Hms_3pBeIdV3(wrOWkRvul(ZIsdrO%$KXDOvcZOCyDQ}ArV|_F_FpUE!=uNW$Q7EN>t_{PMB+6RCN7XE<&#$q>7G3L#6?yNfM}&{rgq=Zk|Uy0RJG<_YH`}qo?q+NLt{KnCqv-a?^`0EWCiUU;?(sCVKpqbzZIj^9~@k5GRGzd;?uD0V!f@Wqu<@HE(%>~@kVXB_=>0b^!PjkkPfN$v)Yj9VW80O+9eBc`P!_%tf>8Moz6`E{)D00m6~wZ8d6f1t4NyU8~9CHJFBxC^{CGaoOnBGmBq7Y^n1=jbJA+(YVG#E&-Vzkl^@FziWy*4ZI-Pt*T$V>4OZZw1*w}RVQH}HR|Z?5nTtt%iEc_j%sWywpB(*uf9%5KUEF?PtwYW>_*wDsKmBGhw~3h{%v-1HTND;01pnD%pC2B?;b*xbjJ`dsE>KIK_Eqgxd&Q@#KJlGH-=^jm>2`BqSWXhH-n1qw8->k3JKgNw{mF1xO)V|ZX4}*I)WbZoiUe79I<D@b1hp~e<RSHjf12KnLu~t%OVyVr<LSk5dh0><(3;)(=}+!ES$MFhPS74E*6d`>Eg4F;hZ{|{5am2=C$PDqiK62G1E^>6$G}@kAo}RKn$kmuXTF8qM=D4!S{5koD#29RZCPYgSyYmf*9@-Y$KGv!^^3%<KcAu_8qKdI6FWOV_NDl|>n*v()l0oHeTl*ibIXt2PO&)FA8QoRPgAr;bhBX{oKVYQw!!GRw*&>Q@%^Cz-r`Of7@ivH5qs@@nrr?d1^GHFa`BH8V2jCY-7oFpUUOW9*P(^Mwzh73r2S~bV$dW-lUdr<@3+&dOZ^~}TeG{omRCtcyynDGAgSgtYXiN<WAH)<O^%;dv>c#>nAzQVII9)g+xxejuD3=FsVRc5m95eFbI4i{cXgM({V+nPuEQm3{_F<dZLKfb^lcwcoMTjfOVoJ$jru>Qn=s{W54?#x6@x}7{0D%y@}k`Mp42q;<ioDkgD1zM+B^c+K`X+68ppi!{r4OD0D5RecZaW)2^M6uGCL-l3JTqKO}pIUE~DhTD(6P?0{<ykkHj5$=<c?De@bd;HZa#-zm^x>X$0Xa0zjbMFQxwU4prT8^k!MixZRc-MDXWQWY(^Uaz^Rn{*c^T`!4$D==bL31NqRhP&=(5jDNj;U?m9v)mwE<ba#DkvUnadbhIzn*`Djtk9J#K@v-@Ur;}^_=#GFc>G;#cfRQQM`q}zZri;p6cg&e}+7ZNGccwHL*E?CdNQYe6-Ksr5_-E`Y0S9!U!E!OsC&$In+SD#DWLpoA8cZ>xjR;URtJX?e@eIZ;=VC>L7k$K7-rK5oA6H2hk00$6aktH?W&So1SJ1LmcFQLGoctt$)upTpMI$h}TJqYhFM{B-oHph(>8i1U^zWIhwRT=8HE~NH2@b-|r?$LA_@9#;!)KR9NEpGrL8&;t&*bLu{cDmj-{#=FzYDvc3~7T9HU*K7j6qO?>qvBO{B~G!rIO*wM_^+vM77r88Z?4-x%X~IN@x{=FJrgih7D3Sf|PgF7?u~S@*HTRO=gYbAds(rwUhNMsk}Nnh?Rrd@=D88ppbtZUC2EA0H`n;x8<=@PkTTc*-J(w0_R1I=YST`o7juh`HQm;<e)o=z6JX~obXoL1$zl1BeJfw;$3IaK&IyuJ@sZ%c_Y=se6JLu6qQO#@3&>@c~91n%0g2`rjRpQr(Ao{TqbR|u|Dv)P6Awv*wJi226H~<hwUUAUK-&DI3yjbvWyGnP@x;nAwKNrLv8*1DOt)~$w%-1q2j%)Z_cD1Y@?Pu5ZqMId23_k%g%9_o|kISPI|w8+gz&81-=kx{3pH?SS_4)vCeE<-i}XX0dbFM`k-35kw^_}Oi!(Ncw0U}lBuuMjr?8nqhq_W=Cn?WOo%$+8~$hiM`0OUd?veXyXVj$=^#S2)SS|l@X_r-;`-B0SYuS_uWrWmCu}mM78P+Fq*t@jbb^Pqk_v65*@kti^nzjj{c6Oe{Vlxuj8@z4V~9tCu`vPvTk3EG#`SS+b<0-P#d8KEa7A5poiOTAj1SgyeNP}^cRxqnhCHY~WY`Ru%Wau|hw7wrdp61ky)hVw7kcqL`L8qsIB9WkzmcT2hdAF7-}?jp3)}k>iSB!-Tj#gfv-|WSquO+;wcGWPh;3%>Uz(4}yNKr`_Fnd8`f9yfvv{qc_55{fU4168X<E~5sUa@4eNZgq@6jvQGm@Vg3w#@X=Je|qILbh>Qn%Faeva1Ji`9VA*o=gA!D{V+08!HF39pWwgjlvYv!fj5wwtq|wMU+p1$^rsY>!&Z@uqn958N)Fx-6T_wa>SUx6pd;n8Ev%ZwB0&oH#*5lE04Y;n^vFM>QAbryBO$!U=WaoPgGiUh9l28u8U_PVNQlooVQ2>GbM5E&Nt*7do&8!X5mSw{@a_*zCoLqd5D|t1;~l=*MjQA{UGL=DnNLch+w<CL8vBP_3bka|x_hl_eFhX}$@ylG%Q<+^u2V$KRmg{49{~$Bd?z!4Dq&A*>rkBA?1V_5#Y_H=4`L)HQ7S(u$KGrB0Wo*+nxc>xf*S7A>Fe4m^A<78uwQvmC+r`r&nYs|vM~^yE*_TckIpt5ywG_{{rT;@*#XM})<W_O!hVwt~C|;qnApM!+rq8ecF))<&J%lIOSv)CamHZ>WI2I7ob8?TV*+6f$p(E`M#7PEsT*@=)Mbk3RB@gcBT>TfxyC_iL5Pec(w@yPxt`t$Vtq=U9^{c$<PfDmvHKgSWXKeIKfP(2jU(>b$y7YJc80>dJmSObBxN#Om(W_>Bto^(?!e2Sx=~Mq93{vsVSBc|R(3t5jLn-h@+pdE?Du&K6Mzjxok(x>FCnD`t_hW<9|{vcvWM&R^N2AmZt6Ava@z)s!M-B|K>Jm|WI*To8XqRCC4F)dtX!zS(JcO}+M~Mco#jeY~E!gX29Zf>;2hNmbU_OB0X8uOkdqCK5sqa$ZNuosR!}M;CYNY>V4{Jh@Ej>zzf({MLV+x9R!{HhIK7ua<oWOIuYh)Lu<SY}dEi=yp0c85<l=ke)7Gm(vMzh$ep+wb`wnr#I&C6jwEOm3<t!)YbjzYW*(NzU@V{&4&+_lX@nF?TI>48B^v%niGb`{3Xlw7U9f$iOq<I9|6#C%75D(%Ps5cIOng5{Q6#Rz*@LY-vLsdb@p+q)(0Nu2JEj%X!=4*W*W9scL35+U4@4VSa_vBUqp+bq~7G=jqdsRF6KXVC_LAS$#~J#tj|{|5E}f>=9RV58s1_2q!c#*Y+0?b(6LDg`kX&fBByb!ba*(GR`hCUkM$R6{#h&DOaF)o*l<)Wc`ve?j@ai<&ZqP91lvSS?fX<`7JV+$EJ+p~E)uKLCywv0S9e`n9v()UxfE?{^Dxd1P8m#_uA|xUG*<#I_|PQbB*()*`)?_GQ}&bPsk>7?2W?Ydo`=1vh6I}+W>XCTM`AF3*n1M)H9a%CJR--hb<o$J5KeywQf*5d$@REpe7GrIUiB%W#BBuIu4+B=i1f{Lw%=r-v%8l5hFb4Bv^r#l#@%kVpjWq4Jjd=gtJJYr?4uP2_!747COsVYyfz;}rAXFePKZK!4h|pz>gNZZ*_K|dYk+RZ<rz8f=JWiScT)CmW_THN8{Bc*e0nSz2{^bx-~RCDy=<KFbM6jyi3PZ&TbKNsT{QJ>-`Huvw=B^^h-;yfo}%ka$zYkuN4~VT>aZ7Bt!c>=Hl4t()!XiM@LE886a4*%B5SXOSI>BlAON~pH)binj}6yBS@;U_MSbwm*Tcp3Qv%&E?Hc4{H{aaA{;H7SYW#x}+H7Y$5Q_2lJXt>Kogu|NyX|rQ(T(!q)GH`!yZoI&X^;$4Cfh`L&Azw8<~^Gqb0Wu%L|KRQ+Unar+`1#3d?K$_y@8<9dP^Xg3?GpKxw%uyh+5vlQ<YIfn?eKF{sF0<i&zq`Y+KMBwrF)GrC+5Y{>Ft&Ht~gY7su&B!go4NJ5rIzF5D+tHnYqKdZqU@j&YZ}mOSjQZSpYfFs<53N0cw5UY;w#FoMQPr-8M(<LXCcELOQbw^)Fy(>HW})vtW?>1t->w7?&amAEtjxN^1WL$y8y&X;!@uwI$nm9-Tk{bBicKJU88&dj^lhg<+we`2DaVd%oHXAC|yN<Fn_Ati!WjW}kF=Vt%*P7~r$R05t$rrQDoz?zQ)Jpy?)VXwSugY0#_bzXhoG9Xg!);|8ZVos-tC>`<zd3jRHX=^zQJEkU8+NvFIoMfh}IyzSxlTne&=fgcc`}hy|Wd5Ef;@ScTzvQM2txk{$%$<_fw}<(D(dqq^_#bvTbT@@8o<CcQ?=>CRJJR`QMTkv5p-ff5kH;xESC@$Rp^M>GI=`#nV!jv+#KDTs?=ta~f1}z5B?7vqPVHD8)z1w<I5$IR(5B-9u)QxXLLX}1DfV_YiLZpTq#fh!j~0(b829G(&2N)Ol`3|$(9fAei>P}qK@B+5&DvMv+kWGQ^a|ovYwdt_{A4i~uvO<3=yHPNYC`T|fT|lCx#qA!SqZ;&n%mE1z*~+mcq!^&qcrM#hGLPOvH79OAoAZP6+M1Z`m`HVBrc9vc+gFJN2_@BJjwpiDh=w%%Bx+M(PAJ*_!kN1qC1;&aoY)|)1QLhKOmQ<i-1&)|9}Wr+iTUU%;@hiDT#x{I_A`4HTcV2yO+*2;gr7|4mY*TFPXAS3h6B4$L3Vnm)n14vYV@QmmrCpiPCy#Ez8ev+V3*)!8G@S!`R{Gr{%UT)-KNSw0?8h@)@KZv)^a$>v>~0g*x@+?9}3h@j+9N%1AV}i&<hW1)kr_bK}nxvsw-Vkh)CA?E{Q9w~b~Qje;lttguSDP&a4SBh^|Z9MDz!Y`i}!gHk}^EmcqB%h92nKZ8D#HH%u%qPJ(Lz{@?fpQnJmJKaSf_0IF9-p55q00i%?6eLRD+~(^jB~yy_<J0E<jon}2qw7}5cr%+66;X~`kK?G%n<wx#o3;K10h|$GoY!Ggq8F@j$88=ug_Jh-76gdPt*|W>teuAwW_V%+keoSg$-^dPsoCzW&+ahEbY#R=PTEGPhj#I0vH}0$!Hh%2@pAhZ?9Pwt{I>=DiTV407`fLEWS80#&%x-eo08FEXwAL7`s^*w=pq`x6BW@08}6(4sJ4!+8_+w#i`fXsfZCP^i9Zk1&-w3V)2Pt4&=Z2o>&JS0poI|_pV??sn%)vN<*+R@qG55I^va{2M&9mz73_{xbTv}rPV2g}knXLdpV{#Vswd0q)g+&Ik2;{&!yWDGv-9ut#ZA}Qy7}^<d4D#)p!`CqMAx!hLhfLH(8Dp_UXzd4o+>pb#Yey1dfxn-t~&JPHQbK-&33!K3X&U=ho&%}N=S89z&+&qXB9C^_k=#)42N>RbI~W^QEFYNXET5n_Au=|x_)eJ>f&8IJ&InmQ^!hj<2%f)|Ax0;<vA|F-a70a=ERd#Jc$5+mOs8|*YjqPlygHL^a1_XD$)_%Fna8;NAzHrC)+lreVy!V-z5Yr%61q$8;7wp!Y7({J-h%JK4{<DQr*nHDAGR%v+e3UtfspO(Ad@FVQ=T>0BFeGN!2^}=Coe7L1+J4-pWRIkIgNr#RH(3-Qf=VgA(&~?Rqmi4{I*$)rmbt5#T*@04D7m+fB^obh^Da=i_r1nKEN2aL4Hq3GJr%S?!6(UVOchYWMZXg-yl;kJ~SRy|we&C0ms2J80pt=5P1==cF$*;pTR4{iIKdn{8}0oNZoIy=Rcl*DR@O_Z6+0^P)(}Ppob}J@4<131@BWQ)A*kpOD6eOZ1w%@ZCVN)@-11n{g21!WwWsud4nuGGiF=2#BAv^Pc>n5Jh#xXy|4OuJtLwKOD#x&&BdjW~;v6Pc(lq9NJfuA2=E8g{=OWDE$#CB<8JOlhNj4pzFa6B38QnE++P_QfcfofLqK@hug>g<}`yXc6RcO*As9XS7+ZVKW;H<ln~Jm<NK9Em-JgTVZKp6reh{KV4Fj;U74=;BX*C2<DRyrXZTH<yu>!xE^h`~86K*C=6U}k$>f+k2K}x6eZVB<oWiXB!=_PGY}$^0Q5?tbfm<XTZS@sN=G!8dwBCMp41nEbV2Rl=J)hmdg9?hg?3U5at~J=xuKN?A9S%1Lt+xM-V^Mte>tl_-_W9XH+@_5!NiI3}7N+Q`nN8fu;N^~uT}f-8IC-4|gE0k<2{h9_ti@0d6w6JXcdB42^eiUmWzcl_7<{;e#{FrK_1Ji?dgelK@%sb6oXL$T?d#sIKG9|gV9opt=)FMzB%WPhYhTP-4OJ`FPP(R4Ure<bO9Jcpu52qiUoa4`{2Fav4fykiJ1s~T-)aIZI)1l%;y2Ki)ZewzSS<nm!%5i}`olrD?aNYblKcjK;GX4LEyOK_Tn*xdVY^#0AoeucZ08M9e%O~zox>btB<`?dV%r1&8^3}Ga4Q?t#e!*aLkLM`6YSih#@Fgqdp+)Hv2;HM8y$|u0IJMs_^uzTGVyWr?szte2ksWH-XA}i)kl*C-Jrs%^N@eP14QRMTsuvhgXXG6X6MHHPaJOS@!BYH&t2;@Cknp{sc31i8QAR;6Ul0QTAo{t^KvYtX@0iDOeetjO0dcGRXf*P*mr$Yv6cTF_0nO{XAeBqts<22D%{nWIDYfRW%GIcwy(zh_R|AL<h>+E$sB<P>gDh_fZZuO+%A!>ao*$~D!WhI4$k+iC;EfH32)y1svU1W?PTA=n0wnv!6XYScQ>!*;@0{C^|>b;cGJ;!jY7A*{7ah_3hG(e_JhJIbaLqC{9>8%0CWK4vr+sr`h{6J8@a^^`^0s!eohI!(^_UsPhCCw+3VX;CbJ$nJ~+(8n3vq(kG7zzF}xdDpA~K4&S*W7Hc3qL1*LC_-qA!G*ir8d;ZWQunYE4gEuqyeYkwx}(8c*KbLo)~gdb*3-11)Yn+X+BpK6pg_*upPKE%A`QdfVF7%)q}ZiMZ%MUUeS_$eHT3K1k_zQrT!&#;r%kmEta=qvcc<JN{YVsOU&T`J2X{D&~}^hq#{PCMY|u%)Et{rDKW?UOVxU&U!gU6OhdH>fFSz~($&Av+qbq6;b<-|mxoHd3d9embF7JN-y6jdFQ+>w<_ohPnWGt^Hb2*5h*=m!5T>&nq^QtlV225c7O8A7w<jf?fZ{)gH_Qvzx&cv?2bP&Y)$xx03nvN!*Xfp+7o;W0%wM_NsAfdV-RkRA|aT8m?Ir?6t1D26&&PA0I4YV=yU?)9WmHD(-r1eO}Sa>RA@e{Es*9_TY3GU)|02uy6NXZ__Ia*j+~~Xc}0f9P+2~#X4U~;>?RGm<xfTah(zU7gX$MLvIhVFz9%VYyHo3Z#VS!q+GqJCv-&ws!HV32Rpa-dU={$-UDHnT|yld(Ni>og-+2ajmEe-jq2Md1F`enxgX+Z#x#LPXQyA^`PjU!>((8;21f}n7J3w#&tFwL&#L|K05#Ulh#w^Q`7+<OGKZsAEI)@ZC%otrv#gSQTejf6KFLrq_ajZ3c;<2EIYZ*}o>cw42X2C$svY%2Z|<jueXpL>g2BAi=T9x7`peHQ{8@`v%)s~C+RuUX@9lcjom1k>ot7Z>NNn_1(J0|M3UsS~IX7~K&nl>-^M^c<^wfh^N5<V%p8<p_T4x<SJ7uN2>s<l)(-RO$yhO0v2=2(_a%kRbyV8rl=1Nf;^t@4Ux(=oARiR30zHR*}5QpyJ&SYzmtsmC;ey}pa<y&rtg#-i0#MFN<+o$y%je14iN8zT_?2gH#b{t7+f3RirTc>|^#Dv@5<U?UFs5=;J%q~JOhq{;!@A+}dZi(G*g1@Bpb8IQwlWvweuJ=0JsUcx<!vUyN3HOAdTj3$oZ?1J*?tcPAowN_;AZthTHG1VTvbaGOGQf>4neWWqRB@J}5f${w0Nm;QU7IxF=$Aitc(w2EK)<o^M}>RvV9g9)o)Co%P@93s^%NNpNT?ErUG{L$9UG+;ZBPTnm2Ws6jM{D!)y0<Cf%6wZc)jGaf8pzW$C}=2vD^u3g7W;GDYz<BtvAL1N_96#MYC=%3!d6Rw^cytp;ZyHjQWWrZ-=ud{Qc_jpDavWV$;;R9@{UODY6tw0XtmHC;Ig1HBT$0imru)e<}Zkb3)m>TXWX~MQCvrM|^0!rS*=jt*8g00cTkUc?rW5xT6PFns7s;w)`97Aij{M+_*#f6Wn&p^<VBD49}C6ft-x!Og|<F3W=Orrb?5wT0&A8ZV}8z*u-jLj^fT%9GANUaQY-`HecTV9Gp0QZxx=2ZK~gGpbot7N+|bg5)N(2uCa(`ts!n_9wJj&6KgP}>h}wnihnJ^W_kG?Pinn6kE97xx39ZZ%>Hv%3G<zfH@*D`wD!-W-A#u~mpTI%U#13yNsbck>L0>I=goiqh5+_cvk7inw1y^C|C;RS`^legwR6<q(JK9OMyA-Y7RhWyu%Dx_Vm!Q}fWn5~5d$Y0o2}!@1~c+1<5N_hFfYx6&q)wBZdnwxORBySfZt{u9T8j*tJQpOohCB1J?_u9db2NyRr6Zl^|OWFM~Ug-wVL+SPwxQvtNt+o&u*yll%Tu*UF_D_$C%1N-_vNNNr#80G#i2j^LWlouh^ivL4t-C<6iGn3fB3u?FIe+4^^(2#iOgA^T)W@!fNN;Y+?NxxI4om;l{@S`tbY%Hrtyu;?}N}_m_pLhR1ZiZ@+wt-`b${=w{1B9@LHQ4a3)~N25cGsM2mdOWo@vD1PtPFzAuUoho?6aiW@z;7z>@(uFlxG<KCHLJzjv)^Z(=<@w>yd4)mz$ri{Z9!8`5-Ix;e65T#=`T}0Z>&F;>1qAg(GtfRf;&g*pt-A|z*k`hzxu=OnYrUT$vs9ABWIbfD>l3Vsu82R`aMQnc&7IDNhu7~_oFn{G&tpJ%rE|M+S~%^59ZDH()Viys99%wbnh3a?!ChwBnQOMcLFU<D{5fJ%F|`e}dBqm5t^^a*`^4R>_rf))RPyCVIV(J<zYOZt^Y2BOiqRec_WRpUP?=kC;O#0n)%YIk*kQlVHq(7riiCLe0G<wCjpwu+P{p|XZHazzYM^NSSvNcHvd&E9J4(&a6W_i!I&V&A{byZtfrU_8e0~p#Hsxcqn7Q0dm>XCR<_#yFyGcVH<9lytXM-M+6$u?**Kwu-CvVYst*_d%8V#^Q>Oo6!&TJJ2mP+H4egpX^LW#Ni$I5Q1EzdB10KP%NB8$H{SHK#-8)ZCkXniy@A(?cqarMwh(^R8qvJPy-DZ#-5W>;cj8o58qq8}eW<0UtTujzC&zhXM-vPV{{{xblhNc$i*xo%TZE~gdTO2B!+{I^urQV3&_I(1cfvApjZ{HLZ?^0t#s$8`Ir@*MAlyTLPnYV6ZcEd5=nw<`tnlno|{@X4u?1T`#g(;9QLZEHWHPy3sDmeRtm{-BAQzn+T30xJ+lrt3+)N6ZhWXAL@K%EzrITayv06ax<*@?1GlugFincd>dBZ-0|gPWSM9$n9?lt))$MvcZ>4E9)(o*O7ZFzWQBaNw$4Eqcq_KY>mo5LWHRF3blFV;3Iu$T{gbOv9-|qwVw0(OPkhvGs&XE0=J2;L2u!5ynfk~QfXwSoVi?oY4sYeU9PJ4!)3IA_fnhL8)7o5!;Svka!Ae={@&yU+a1MCFr-+S<j)ZdDB+$QXVSukHxHFLsiRJ>Jid6FdRX3qnij&2GzpM1V6R71Wm)yB#u-4BSrcr!cg^28$X9x!bmMCxiF_Ep{{|YLgXw1T8g^YmTX)No%7P!2rz<PjPG@M-<+;H9;`>LvfR3cW*Tc^~yN4U9Zi8WCDgRHOan)9<Y#gyvI2|q_6nVD*W{271b-dS>e9LS%kK!old31uWoj3Yuqu<tSv1R7d?>gA+znMxXdijclIgA39HgH&B<9$hGR<P~o%%^5c;Wu2RjZm&6qmtADdt)oEW}SMeI%2=*-@x!Q%(7fdi*Xx&9_xL%zlIWbpfAwgmI!lasMvHV_N=rXUwYe7J2mSqxV5^s@1`}9&+A%y0b<;RbdSk?J7m76v)H=#Dq7_c(c_76sLXl%*UctxdiM}tndN~8^HSLPtSmx&pmeV%pi0>3&r7`<G`K$(=A;~)twr9Qzqv(6so%o&Oh5%{UWuBiYlG!Zi?HWqVOW}$P9#z3pg$?%4=5tjGq&^ENHc>szQ-`6eHV*WO-K!-#Rv;sB4-V3>GWzPC~58?&Me2OEj>~al6R$q0l2pc^u<H_)0#djD(AzxgYV#|p*FE+(Z|Kq|8-A&?N`v}AH{V`S%a@JjrxnYI%qGef3JFHeoNgV+LiY~ZXend-RGTAsEfFt+d7AM%|mtrI_JxiQChY28`Y(f*26l(Z9RraGitr+FBCOO&FaGaNx1pIlMx9jmxYJRZ1V@MOQ*i@d#cDAYf^Z&INz$P5UBSKM`;gL6DCQC_d?F9-KFu6sP*WS3jyDbD~J9fB~*BinIE9dfC7%Gkri64p5K&#k%5N@O7=Qi6yHwQPLafaW*a;jHyL%8y4B(_c$!Alhggt}#w2qX%F275?%!WWhN5c@Z_byjX`)iDPOe1>NFy(F!muwTJrEQzc{*@8<)(~ETVPu$eX7(W8tnhX?-;J9WHS}y=Ys3peOjUCXiCf;<Av!W*!?wcw41a&IQh5c(YiWmL!wV_q16OGIf}@9Z=HQkA=WoCeR<{T2R#Qio3_T&9VODpi7)T<;`ag&8ry7--9f3{{+rAX-N%jE=-6g9eAA5GATy*`DkpEYdAXC`#=xFn>>cp-dFyOKT3YmB^|H{-xQuQwe;(!7x=w*Z{s*glt#M;yE~a<iPypv*vKew&dKwtAzr*aC`l(+t+!cYP^TA1N_l2t3t}#~IS(`VG-KqP8j7zPe71Onqs-MNRy9ny7o>n?ssY8!XQ9nM%Zhw%y?Ikfd12?jWFR$L5a01|n<;9@%s|~AqD&75^9E-(H7OK*s=c^_da%RanAhuS{%XXDQKeG|aXbo)K+WB_O(#k_lktd#*<Ax9~3rdly*IvK$1Gdk{y{UfSytqZ?voC_U`TQLcbF%SfR7tm$L`ike?sim0Y@BMXy*8!n{j}JZgdd`9y+7^kdYa@(>D0Aza6DW%eavT~Qn?SZyVYWe%P*OePsFx8Ooo{}09I#&`hHc={q5U4bbYcu>z%}um}aZ9zDnQYAqn@L0kP3ADeU8O^rdE@<oLID(^-!DRR4{0gJB6im6=j!7Er{lYMuPg36D&CX6nWBOi8!V^e<0Bdym-`gvYHXr|NjN_Qqtpc{aBF%_*IFbq%u`Ym`DOeWgvA#D4zmSS`0VM2F|T{u~d$)ICr|<)Kd4yPjRuzrKUQ-y0|u#~RY)K7Gb|De*i3ZxLv@%=hAK;|_TJiXOUzOYg?a$cYVkCYVHqN3MGse;Ps^#I<cl$8hzY5pvG;>{_?AQC`PYao<x)^%R$X32u;1^>xj9kBGb8s)oSi#*WqYg#q8E>K!qM!bCI-V{OmhwQYSlEbESeK1+|eJu0r+TNUS%#UrI|oBqam*abax#4RteV`%c{_+5#Q5;dH?8Y6?fI9*w>tDW#(@u&F+OFepn$&}Y<gWM;z`^hx}#>99&x|OM(#tZ1i+S#1Bo3+>-+B7L*k(bsvMepQdON|ROs&-fI#rYMa%iq}-1|}<SC6u$(Wr@&LL>e2yiEecEY%roa`(sZF`>-C)0{gx(fpYCSo!z9mE)9Na;eZ=G`s$5NkZc!pu<l>ZKHi>;O5r8g)mKYI5m=MymosCx#jcIEw6=~C@a|uf;W+JE!*bk~6|Cv5oW{8P`BU9Y25zs0aXGneVF{;BGWiButtPg*y~|aZcgi9jT~<4!>XSsD@aoEwdu~1&i<6WJZ`|GU^^znF^{wnVz2A<DJylaJgO~{P9by1e)@c4$;=BzBb+8y7>^vG?6C^f6cGR0+=h*>%J?9%^5<`CyM^TsuCY2TjTSBi;&^tqXQ{L&@RsMQi1jAnZ<o~cw#SKoR8LU_jsn<8IenhIlUv4{kcRI8hD_Nar*JWHKHRu%NaVii~xHcT1vp;}ZcCW|b>s~%-<*+wftnkB6<z(a}<XR)M4ilx-wnn80-i}U~7Wu(<F|Uuv%dB^w%UMqxcEwj14H|%#KF%|a${Ogr{R%F%KH|ZXc-ib+8riRY-LQi_1}O7MlYB#5M<1D|qRCI(n)2|_;b<No_SnGJ?a8jU@VFddHqt_#a?@l}?$`UhHhT6Kqf-h*#(<fv%gaSgBI?rtT&45;aWk1=<F~;24_}G%ojsZ^jcY_6tK(f{JMfe-Cr`a31f{_=?UFJv%RIi~_&HE}&EF?pYW6cEuM|;#^)#01pRBtB8pp;_e`#s3K+G{xV*nCNLw(01#@rf+@&}t6x;Qe|-!)v*vL1lvRM<Uj2Wsb5&prG>e>n0=N-F|L0NqiV-v2Rm?M>&pU-z{bG}4$65hE2*Iw>X8A04C|lTwn?Ykz!Zez))Q0`8k~U3>4f_gZUrs;<$Wmd-T(v!RW5H~cP)d0&a|@0XbyT`c0RZN#PHW1Nk(88a7-OQlNh{jk+!E5h~Vdxdlo;lKMp8G|iv?XYT5JU=>|*>*3~OY=hxS#61XGl!6yJL2DVP9?L&46gN^<N=OnUDw~UE(Si8{h-_#T`vndT>w9iLxn4dM|aq_pgzjZDxL6Y;OZf}+1socy`ZM+j6m$4+3fVSe%;q4D#mZE(;j}2gj<zwr?-nXQRBjwoLQ<iezaA8so9VrdfX^!?|K>*Xxf43uN7IVj|1U+;wTiH(JGnk9>uMS3vbfv*B6>ew5sz`do)iPosx0r<7;shOF6s=1k8-QfLd%F>BrFh97;Xs(tU#V1_g-|vCGUW(Qo+D(b^P~s#iJfmaA)R$=D6<i@EV@gTD=(7-I*M#dT>tJJ)qO?LN?C>#h0agfExTHOA*iSwq=J$#|ao$3F?Gf{fVGvX0vhn;rT&V}I@@XbjwD(!-)F(iV>9#mO5_A*Sz)d@Dc_b45-LG$tM&yLR`&+i3A#p-`V*4{KcuPwJ~;cq>kW&MzoDrty+6%k_u@H%EBhZhI93H&3tq;)HE)jY{R4C5NZu_~t~}DtTKgj0ALeqF4iWp{v0<J>9e}-O#G@WjOj4$Mkq+s=Pnu=&ph(*+a3Wf8s%Wc|q60x~E3MlJG5^?`lgGHLP^a&6*qyUPY|?ekb+G*?iW#n`{2sWFRL@7F4BqZWWf`sbpO8vsn=Ov`=IQ`Y^yHH|i0(#8${D1&W-jxq#zT&bE4T*qzz-ekRMkW5drs!F214w@(q<Ex`hFTCGrEfRd)PKxDMPMn~A}TTv33DY#e)#CR2OvqcmdrN!e|D640u+@S+_#H*t)$HPmz%pc*ad5I#(-+@TsPh8R3^3Sv0y_z-n&>2lfl!PU3$mBcUw1rlAh@bkkZuH9Wbh29ZFU*StDOh3?vkKcP;2Oo>w=N^SptpIxbsRsOhjLHX@=g~X6fm|~su*36YGva^HSW-C_dKdmi#rSE$3st(zycr~s)gI8dYjh|az@78a-)9rbq#;C_OHNt6N9``CqGx<e;>s+szC6wq$$$pcGVeOX-z11lyH5#7(7267Bzs|ESm?8ns0))iJ))+o>o!zwk<Co7QUSy{X(O8fmZbyT8!jfScI*@XYKA(Ni2KghU^sLTX|>!&{6r*4<=fqUsrQuTQtk)UU@XbLOluH=lT8-OTl!vffsH0(CJ3PIfzfyuqQxO$_*%TB>4w}+G_;CoOxWTXSkQ2utaZEmI{FvQ?|U!5@-Zk>|axS)BvA5d||NlQ;pgV7mM4dL}6)r=}bAgT4$r-n=W+r?jskbN<Q%~X9}v_8dMeRt6ra(Ha-QivCvsz+Ik%$uBa~iBhy&0KMvXq^q5!q8{M*^8WmX8VK7CS5#>sKa^JaFcvJQ-_N?Kz^BEnBr}A;wkr}m%7;H*^v^hf!ehn#WDX8n!9kZkle+#G36SskzySR9Dr?QzK<JICY`~0?xEVaE?=E)biJ{<*r6Pagxx(d2EM=X0YmrzlbN#flq&6(nbNQUY%lIu^y>@$a<mi8^_w1D9Gy~V}pIVYcjzN%mU3ID)RB_TKELe$nbf2c)FzE|jg8DQfz5*~w5lBJl7pLdx3!qQF+Vxdw^T$F15Wf5v^xoTdZt`-D%rZi=3KX3kwEB3N`X9`;T;(IjpT2D86e37Sd|9(%K1fBM&QGRh2yLj|c)}1qH$4|A=_e;PqPVkeLw2Pn02K1Jck!@#Xy-{67WFW@OUgM4NN^hRsboLicH!de-?7DKhzie-|DPJ(UxTCG(q_!UfOtHG2t!+(tpQYK6T2%NPU7r#3%`Q&_eP!C+Yfn$ll+q1jZBv>}CQ9GG(~*21|64Mw68|^_`QJRaY95~Z1F~I*?TcsZ$)g7pD_CWEEMEHLCF`(nPWawwBWxU}*voyEy`g?tr7V3SjbjJh6D^i{V)HCH6g^L!k(W2nSrpc5mFkZwmwM?j0ROm9C3QDfw2$)f*6YJOD>QL?U!7_7y_#e0LoVrwp$>1fpTBbESW23XIASMNtIp!;YI>VB6EUA3`W13+%j&s27`9fiw(XA}NVm*yllf^Rox{Ikpj?{;lYJua8^zMy<_B|vad_zj$$~x2=4GDIPX|Qlpd}PuUWljC78};5#&JJu=|R^P12oxF?&BWIg9=li%D28idh${)aRjx)Dg)xMog2iYL>{s(Dg~Ji$eOs~sNi&s^i5rS*g(-;al=rXFo#z^@*s1vG1x`x=rHs6{imyoyb*so>^8;Aj=-MJ#g%Aidl)vhz-f429!S4lIpN5zMpM(+iKvSQN4WA@YCc{lMgWiGC3<)J)$~_rUff%TVvlp%!Y{xE*)4nMfleQmska0E$;1h~>9-`l{LUGDMP8T4Y>aQ=u&)>4>5s`yQ^o$WxW4=Mdb$6~G_vw3FU+034kEC7(UrvWxT##fHt01HZx)Ybw>0@s&Bs<lHk)&WPlS%NB<h916FM&v-+X2hpFF+CKrg}yw~nzCw?8xVX~~T;45{6F`-)$s#ZA>QI?CSlrKNMC2ZvocE|SsM_{k^GU3v@gQHhS+@j09Da$YtBXkMX^Wf8ePXi5FJ(DHDpIG^QoRbJ^PTK&)?Qkd`Qr;^6<`S|hCS7t1B)l1Y40j+dfNaOcRnK}q%M(M8bH!m+97K5lEJz^%q$be!&n_LWue$&!_j!jwTt;F-NH>kkMb8*xb&!;wf^>+Q{y>u_ZzP)OFsL7Qda!QBPw!J5Jx1H}L;WJ{pbXy9sD}76T&ILgbwfU$C^KK<B6^9pLnuGEA8;73JUo*ZhLA#^x1KUX=Wk{=|r=xtLt&KOVYiAW4cT{J~Gd}3WbNM|w4hEbG%!OBZsR6W(6-!sJ=558nRqJD~WjD;u`^#W>k80bX66JSAw1U3}XNwe&3^rTqepvN7&%9$}kmbbuWV@F#o%`&;<QOGjRpHk)=vX+EcZuJw8HK<|Z}5=gm@NvEO?+6`%|AwAH{7@HuxdB#=6*63dH<a+etN^&TF;MocjC<EntM9mh1Jk;!OgO`znJVQ-bEtP(vGVLOQ#mgkJnb;I`lh*vhwtM&pz9aQY<RGZ!8up4Jc?ZJ#Dh;yyTwsnL`~qDOx|@{ovZIv@VWwFW@gPL9v2~_bzyFjxQM6)5L{QV=|xCmrDU8=eh#hfqJam9-{osQ(dLMFK~Whwu9|e=?%86*0I$GTt^nx_QKoswPa~U?4RWcH`46&PO6u0&v4Yc&H+sD;Kv)(16FSY$KyymeO5H}ER~iwezrd$epE`4$5F<$cp0=CGJ<MPuGj_yXJ~g>^HXm(XDayVjjlu?Y6Q(^^-o2aS2wHUez~3gyt&b|j<d7RP>npiX{W==^&Yl6{-Dl_c9F;Pdz^Z$)g-Wi$MAO420NfPTvQalHAdPZ-hAS{ytJ8&b)!sd+)1v_jp{HuqTR+sdyO}y+zdPHU%&}!vXv0TqH;Cwm0CwXT;SyO6FTX^O~v#Hw7*>{r!%zPMoq(B4PnTCVVa-4D~SLuLVtW4Qp97f7Ad(3_e}sUas#+h8YGR&Ozu_zZs^^2#0dT^K?$7+ol*<(;RIn|$*uN&bMdnr)oCjr%7V6<D|7X-+9;X2pb9sMxL5ugdhX0~$d?EgL)a^;s|ec!s&JY&ce}}rm{rylrTQvPg8id&c(`*p;Fj0h^ycpB&~s83iDN?r;91cI%x*Gz;d1fa>6OnxHbP~!oKcFDh{zwct0*237k~fUW9)Ky?Qp1YAHcI6f=Ng#8i?ci36G(=@ERVgYqdCLPkK<H%Y?-*kVdC>{er^!!a3dMD|(sL8FwEwORm~264%Erv?1k5d{zrtdC;-P?Ph<CfBRc>PKvf9t+N&S?vwe@ih{vIoK=O!Tidt9^}TK1Ezw6f9OK@j>Y=v{0>0yUN;etYp2yk6t-h?@d{8JoT3gnwZg`t&cbwZjqFnk=zAJyTU9r5Ssv<{BR<nUM_i31)pmPMp7WF%=3RI=bp@*~ETob7r4lc-n(<kfu(Cfv7M4yvGui;$i!-gzF{p}A-1>MWCkd%2dpZB53h?w>dEm7yX+Ij_1ic^-MVM$l^IX(JKRaTAV7v&4_-5Z_ASbPDXW2x+e)0jD#+1AOOv@{F5f*t-?=>D_aa3Ou$_7+hLc=I;5JPil*>*4Ck8*3iSu<zxCUaK<}9j-v?<0oxd@}GQB;7|1M@TgF1cXS#`)(HIZU7yB2kFwc>AeF;rqiIRUmQzSrgykpPLB`@*^{@F(Dw{$p13LbA;aCFMf(OqlBb%9;f`kj*i#O@5Mr9mlOa5Zh*lM%9vRO>pk9s8;8Gnlr(nA?ucPjmlh<JsdF?P5y!%}89z1@tJnb0vcYsmyur;n?XUhBzcX2i8}HX}+$d-r+cxLeR$C*=LAc;#c6z(kGfxA^>2kuzok3cnO*fMR3jBd$kE62Q}20A&tRWkzaC2cY8)_Zn^Gw8u4%&WuD)yX}1+ygIsDp)P{`zQE@?mDmFC{Qh)}T{IE*8rVoq)XHe>4e4~^_!0!Q|MyYj(k7-@0fC)Hs}5xsa_4P#L$Ej3RlXUnaE0hKeK}i@6QE<4=jk19o*tEvHCmhntTV$uBX~CMQf{^~FN_<x%RGkKjbhr_KaMbTbftRhkK$gFmfxe%9+i!;jWzBfViW|8TJ4!tW3~hOmc8~y=Ajkv0KaM*I|uJb5BO1&Mz_}DshWP5?RT2OOFzF(w9+M|jy@q@Mh!$&Y`18<iR*94$^)xmBeqjxj;K;Q@)=|78A|V_(vD_A36bqyTnRU`^0lCur|_C&pA*^Ghx2u`lto3*^n0<Zl=t96IWRvR!+rR%8d?cG&ij31IKp^3czz!cM)&mT-bl)W^2PcHbGd}YKla$c9o<UP`q>?HyR*uG{O9%m5`CQ<-;(F;;yU-tm8*i@9!*@V+(@B%+#3Yw59i%Z^69{EQQkJMs!2yo9^xf4%q(zS9-138DzrZI@+eg8>dsc;D^S_3Y+j<38vQ6pB$$$Hygz)eL!1=wx#sg|Ls(Q!i(y#a4m^X_;q4z$3sQK4!=jRTRGCQvfSsbVf&T0Foq&)Wpd6r))3nnjX7lhV-{Lr<Ue;5QNh(|bL34E3L&Z(=S$f}?`?XJtU>8-Y%h9j|2lHD3x6j4`iW<EQ-Hxc*<#3UhiikC(#4*O(zwT{!?h5j<w`x@f#YtZBDCq9n3Ja%x>^|rxyPJ*dRdX`3eY+D#=RH;AR?F;SLv!hDX2!<xP7@Gz&}9@}mBMv5XL>1q==V2meb#RGbNMFykyF*Nc|{8hP>m;1b*l_cs9g|I^se%S#%yT6SS9MmSK4xU;S3N<zS1NnM2v8Acb`~!Wg%zRL3cSCpG2+yZlmm=A>f!b;O=kou$Jx01Wx|kWfsnh4<uE%5=5iG^@;oVR@_`BZ$j+7Tk-s;3z-#KtI`2K$8YI*-ITHJU^4-x4Q>XC-8~z1t_vmzp1CFvt3laIn*!uA<tuX|J{l2Y@73QBgv`|l?yK|7@<QvN)N|yB?CPhgxw6spq}Ddny0_E37s-DhLSo%U9n03;_D{8nmXo{q`X0P%J0YSvO~B71J95hk+<u>((qaeqBX^D>ZHl-3{hQP{|8BYIfRxBU@#t(n?Y)Xe?A<>3lu}EM2)ChmiX=O|F7Jk2pd38>zw=G*XPH5hEAIGQ&vg!37b}|MYy0~bO0%ohw|0Br)6E?^q}Mo-0FOGM`DxJhZ_aQUHTJA?0EKqJ43hh4fKK$w7)`3sp0ovUuj3%g_SrvGN?9%GnIIdT>MSBAL+d)+-Hr9x9yqPAlL=9Ss&%+B`M%DJ{lSlz){_#r%J~oB*0wYMvt@RpQ)*E74m!1oK3qyzIY6T3SZG%dkV>}tb;bnZd2^UpkEbcERbcPWe3B%M1qNNt_Zf+|&L;wkW=P!;Tdq~*nt$ODv>WtBXK%&1eV@{==lHRM6e^w#*BXz$pbZSDHwcv-s(#p6^j@g$ox>`hj_Wf3Op@t4X(eaETmT@~`X$F(K0Mw8QFg6f`v5($Psx>u<_MYMAF)FejMgT5AyTD%SQkW99QSP(f~Lx@;R{gfhLrEyH=vhqZ15hGM6)tlY@1_JXqWirQ(!k<DJ7op-=eh@Go+W}R=sc@T&~^Gjai-*<B4N9&;;*x&A!VPp>!_Q@wR{K_m-dEV2E8!=a|)t<kNDmKVUfQN^^SqXkCu=E_Fp+FM-h!bzfhao2b;^Jf4KbWZIqUn_2IX$-7FU@>T+j_aQV3fux8zz=c$lJ=pyukD+w(r+>}_&o(R7CZp02J>zgI`BcqORMwlwY?^mw`@pQgPv&Xb`JM$#W0V%E{>(uyq_j2S0+`LmigQHCm$=MEs!DF=?UVc5JylY?aoq_2W~5Le|Jb~X(SH4cEBmIdJ&fZ3za+b+MXj!KBIJ>}bFDAp>0a<}xmyUXFJ{x=g$6OXJfL!&q9ye1xVv*s{166a5rWiJ>2{^Zv}R%1acwBTfai6_YQl|*Ur5gkANN=5oM5NvdIq`%d~x=A^^26|3!sig^>g;5HKnj^g}dXUinW0bXhyVq*cpH_EK$9`T{d&sa}_XFOI<)w`@t$)K`hbeT@1=G^)mMX2ZS~)`=+1rYFSndY1WvlfxzSY+ps$R3G2&f)&@rjvZq^*hjGozfGSlG>GC<t;mEjW7brj6QhQ3aeWTH@P}96LMLXFxE!q6(Id@j$z28kksq>5zpjF5bt2)rxgqabT)0m#yjZZ87Y3E)?;d*?)pL&VLIpSR!!!!HEw?E@1nw&}l<2KtK(<s{)rNiU^H4$bO_M+KiP)-{KT%MK7+7GQ*_<29`8?F6OUoSlN{0-LY<gy3!E9kQJx()<kBN{ig_Ws`vt&hsQrmJ55aXykJD|H@0H_C>A=7qDWC(!p!_HABBx71s&l-NSwy=8?R{lAa8JmQ%jY8s$Zu~O?Q`JKzg0TmnSqe%{HZ5QQ*c6=?9j>hfmTDYXr+hqB<*Nvf|6p${_`-}&J2l^Wy!t3+xiySI-l)L3}!-J>oZ^?37RwY}2VRRlWJSK0Sd-v=)(bb9LFoUIU(VeZbnUL+}<$ei`X`A}-5XdJu6sMbSHVVufAj`(92XBx|KLi$^Roys&XDWq;cWbtg;JoLwoNyD*yuxz698IzgD)6O5=m<(!te&6F&Td-k#brx~RmOem=5S+_yXhYX<>Ixk$!Gt;0XaQkwf<lq3~KOIjAdeJ2%#eGx>|3UG+ep)E>Y*ns25*9tpx6iE@_FkyY6~-It5o^a}v$#?V$B<1>}Ld{32mP+kTqClqoD~$>BC!lwXgRDpKcFp){CIbYk1DHqG`oV)+dDT2ks1N`u|B{D7*Z!Pp(OH$87wJQb6St$V~yx2i{izW0*-AtTNFRAVXeCj+iek*9nlp2yFKnoRTqbLAeyl-b^QX-Qvx4$lwzrWXZ$L|m5W*6J`Q)<`SHfpy1G!jrbr@rJ8wX;V(<4)Ci#jT80Hhl7A%S(LcZES{c=6P3%5;R@+Bp07fEiJfmytZ;t)GdORU{Dw9w^!T-&x(S^$+dS`HUsWqAY-#OyL4`u$XQ3u#G<ID&Mk;@zZ0_|L|JoGp3G1idsMc)2+7TF6>J$M#w$uNbmL55KFV1io`GAEZTdtm<Emhp%(H3A#YH^eJx0F@DfpZpRfFUl0t2)Z^Cc-!C7v|m?{RUZ^5@+6JQ0Pq5$_3H8hnqCFu8CZ6iwB@jHLKUhfskdfvc|prs*MZFn)ClwvXEP?1gx;$b-iqLhK%J}4e!|nqvK>~7U%kDyp6j~3mOI83KrA8H}9WmH0{oWwVz4P<umP+SL<fumUSH&M$WNvBP-U>*x}GTmi=c?!v93<;WEnfwYbChYcyw??nD98ZwTPH(chP87ha8)Aim=}%2J)rVUpsx)9f!Tj$N)0BSzo^6US9A@fjP*>SV|YtsJ`{Z|81oKk#z@&l6;X?e0tM2{dGAwYO6YIX5O+z3)5`WG0$_Z2Ndqsa0R>$7C_!G8W9Vf}iC^sW*h2YtZI_(v04=1SItzkJkEnIRAGMf^*X2KpNdt<^F6wP<tLa5+&aXn8|^i$=vyM&!aptdnQSBA$WBqE&1ZTM`;lo!h3nPSDo|eSabOIrCaP=2HE8}h1a8|V{XqsZO*BFB%N43i8^&kN&nORGzYhlk2c%tJy8uEhWf)8=mQ$l|DN4;fgg;~v9X)lUY&{S6}3MqD6gsbnt%aP;9_3Bj0=2*b6`)24pD!Mnv1{nSy>Ocx91fG(cX`I>{r_y3)%$%tgrcZC2EY^`}As+phe~CBMra=cl3U2F8hafvf9bBKt8l*(ZSVP<jCV@1$4aK>6hl?{c|sg;5>YOkP1o|QM)nt*qf%mZFzdbEY;rCEfS_}sj=>NwrWkcWKFS)couk;_9G*|o9`xd%AVhYep_`WWc6LgX3^f#uM7R0TPwo=#{8ZN>m|YTX>RXE-XqRKjwL9Mzq^lQ9@z5-dQ>PAIWpGqH;Fx%>Xn_)>64>nT<J1BVX-J>9w0@YQ%*hu_=;RLQ}I1A*$)n>C;)HVI7*`MOz{-~h1ij(v`Q7Timv@pPq81MZx+Vlc}Z?p*WThbFR%ZGJjEPb=c{xR)%)DpvVq4sy-wEcqM=e!#o8LTXY=C=J(Nu!U+|TF5ji?>E9Z_=%k~y))hB`6yx_lf(-sUT&!ZU^BB%WrW)t<f;8x{j%)W<IVAgXLsqgTSf1n|`XpCsbjMlB`HDr~>s|+7|zaA9Ro1LM%WqGR)(O*gTrpt9PF#Rm_Q@GmUY2naq0!C8=7g7UhZAdf?{p4W3Q@{vdMhYLhdynKwTH+UR)4Q;b2X;vZ>q5f(zN&dB0@<=p@3)(0@qHZ^uJ@I{b!Z1)4vG|4Jr9TTkoplpeDt*hr$Bc!*cMBMSFpsna&MTE&rCPIPif2!TfRM{%0vk!jF!drZxiJd7YB1;;cek>!cyz6F})nFfW-#MS9+z_D*r^r;e!!8tgt(kZo^J?B2cJ%IMrOM`?^?q9Y=LF#rqedB37F1wzZDCn0KIS&qBH+M#&8txcT9cKWl|i6C`^17jqteS*tj7E(TZhBCE2}u4=zi-WA+`)9oN|uDuY2$DuMTtQh7aluybO*M@N^JN<^tuh*--eOy-uCO@Pkkh^)Wyp(%Bc>7{7L#VH1j5U@MvpjX6=DLLR#G-oJ%ZBl4J8gFjO|b6O9e(Ufr0k2W7+S`=xOl>fNY@?4zhGagQ1@r9-vO#_w+?~%>?Tf!Q|Q}dxLP&`A-;kA_N8c6%m&e*uIy%t(7>QnZx2^hd#}6DJs!8w;^)p?S9jy;Dy&9Ae1A`9+3r~VU0CZg?RZF84<<(j@J!|wH_RV>>U?xk(40`>_wBQ=<N?X+Ty|CiBp{}oe29ed`b3=0dP5at4h5f^a=3ZuZYMvwD11jA9Fa7qF+FaAd%^f=qYGG4c~)6Ft$^*pNJ{I5QOSj1+4u17an8N0N%7L2-lWqhDKr9|epQwf@DeYBQQ8w|wii+NoASh3^GWLpIzXd*bK_niiRV+A-j%qgj`#mMEW`3^D&e()e@BL*UmQi<jpCW5Z|P7Vc#)s<rj)RH+|QZI5xa*GAn6_X4d&I7bv$n1Fm$-&k}Mkai#4B?)a?>#^D4#jm65cz2X3n$vdgI<1$RRY7346q#Z?G~1F2eiyK!kY;n4>0sDK@H?FCQnR(DiGsN`rWCWpUKZS7FF_hwj1J(6!lbqVvR=XSnjSl%&TFmyl&gY@68T=$@&r6FT(V$rodGk&P}+O>m0#d<R3NLE+uYSs4!dIU&{{<$}ZwHy@tc5$)#^#9C@@agMu9hL)QOK-GrXBO2oaIMb`^sa6G)X;bLV~v65*hhbr`Ja0o(><Dt^~sW6rD{krkN~k^{$w5pl5C8_?dFs3;tA0@kJ(~G77WDN)s0n`x=rrmpRWJra3p0l`-HVRtch3z*JEBCFc=3)`AOLKS<9&&=HT=UW{2e7BF<?x__T_0QHXw5cjv5TLB(x650S+8*Np2C?Kjr4v7XXy)#rPXQfygk)Oq~+;f-K`tINGLww7Wz8F)LR5HK~#0o3Ac=8RSH6I4qdTOz4Z$$|yPo~$#;{L=)}`-{<vJF1Ji!~Xh$^60x&w=g#q(*pr^!Pl#~-*xQ9=Z%&6N_Z~G!P__n{JXts5RcP;CRn%1=6NbObvDYqQ+5e}5Z;%@hsn(vMw9j!NwM<_IIO7-JR8=z5<l$q8^evV{TW_MJHStSm2WB%#>h#oL!HXYH3Qk)dZYP^Dh|=d!o+T~1NVOW=Y>}y`rCtB^B7y9L@&o13^9SVqEopI$-P#G7jkp^pwV(7`O3>_%S}J{Ta!^`-Qyg2mv*;ss)%J`vK*f5#iU&X3h^z%*?I3-wQv=WaPrU&B(l2?7H;^)8_=v&6}b&uJQm*W6uoQ;esSd=mYI`yuhwjRIhG7+tapaPzh$mj*v=IFdd-YV<t5_ZLL?xV^GO<jnu-{-V7;vIr>`JOGJNTcO}((Hp-)DuPPn66S~#Z39~xtFV59*!8Z?m;3>d66{Hc|I;$<@050Kfsx1M|vIBcX_Pjhls2S|<vDZQxs33%DWzEZ6v{<nT>E92HTy_b8dT1?|>eUEEVODfE+mnEJ4SK%a`xxYP<4=Zn8{P@jJU(9NJvLOm&K60KommZv(S#v9%e;!Ua)7vwG%`5M8&0=6nUF-8|btD{_-Yec7?`S%orG<uV_i!^jz~5I_h3wBpYhuh*lQ+Z1y!;yXx;J~iE0}5mF}sAvmdSbmiPtv#o{JytucwjBw5DC;w7pVM$W5I2JLYVRVyi?@FBLB}9_JtaN3A6_JRZBm%=+e}J<OI~lLBmL!OGBhPrW{}(V}`917<MyqPUVu*y{WK4y24W-nR}grkiYkp=SCPyb&?9ZjDO&a%`WXZn3&tEI6M8E`J*us>*Qkff+~!zJIvmYZEtgONe{e$wXdW_ehMx@O6XKkme0t6<5pAVwwfwmFTRH(Tv-4q>C<zu!o+oyifyXX#+7gh2NrbYj-_s1k+ANN_|^0J1wCyMWnhDA*|LrRHL17ML+fX9m?Q|**!#$PJeRKrS2VfE7>AdQuDZPC~dH*t%dhOe8G*Y{UZ@9#hZpJ8|L_KhZ1pt1$)j;4me}&MvL_{+V5_i{SN7_*5Sr3FJKrsPT6IFh6mI)0Q1J`Vd3Qxs|Ltwe^ESjzWxGZX2>myTAhbQ;sey}zldXu7X91O&j%j++Hy;px7_zlxHlF(8&Yd@RleVwTOy7*MQ{>R9e;9^dHCha3L7@uO8uP-nQ6B21bq4l2hZOVa96TSRLs`Ev2MLHjAlD|+eMOYYkLFBv(->svupmkdz7osC2h(Ccot|^Z0`2mw`rJg77l~NWK+dh4+F1Y>)v5J;#LhydSAdsAq}(o;%_RQHrbYRoYwuR2eM7K;TNjI#_+lQo@BiK8Mi&BH&UvT+l3Pq)ESGt!v;}X=l(VCep*JzioyWD;wm7w5vfssY<|^Hd2Quk<>Q|;Xa3JN^Ib_~D(d!vc9ojg<*Yv*$aOOkl7_vgJWM)zgS-Ms%&uqzKByX3%h`^d#<z&lb8g4NJG0-jHxiadMaD-z?ZLqRgT6qA<Yn2^Xwjg6WI7-8fDOL}24r#gLhh>xv@XX@1huN!^}glIMNbvd)m5!RB6d7bdUh$^kKy91gN1@saw^P|pPYAo|5<qUL9^oZPr><bMREHwux4%(ZrAsfU~Dv_p(H=cV{qTIxJElC^sVKJFM7L(i4{xtX22Yq?E3VB4vE<?jKylKhY$JHFbkkmt&{_`cRFj6buYZ&>dmGR;Q0QNvLfA9v{a);2H>47+3e0M!j5&stLbGy)iq|GC(%e3V7H~k=)J`F{d}Q5cG;lWDn0rd%D|gK=Eixs_&j1W^1T*Wmw08U(fVvAwiKFkZS!ZUljrM=RZYW@nK~ov8fo;q^7N@jjanEMnorxzC2JrYT%Np^lkL_s*(u`J%IX!J3FUf1|3Wyus@|t*SZGS6F2U@=Q%MR3Rypuqpa!6k(VG&?wXrBCC!3{QSyHJI)o4#o>uXONP=(c^c6Ys*^IYtx5L-7Q%IfY-JtAM>HG5RJF9`W5eNiyzL;r&Fpc~6zT)ms`^Z4QzfK+^|F+|-e%pNQ!47qA^_w{}bru6BKOUwPZ+{sqy!*H%{yY&*P<?kger{`j6(}=FT+DvB@5@NIMJNnz;>=lk1WCC~amT`WqACuLrL&8Ck6NCLiBPNi#HMU;=&`ulFowsVux%jHw=7$f|aBd^uAv~RMHT#13dD4+M4k27<3=W(?cBXD^7Z4r!3jcl7(<v@DvQK})!Z6Nbv{heH`$B8TV+Ty0KNO(9x{e2fC^!xlQwM$9M><sTA^VIk`}z107UeUs?mBTxRbQl3Fw*3rNZB%BRtwj)&4E1G{H?&gl&N6dUlr%4rrj5|i{5~*X7YwTc!o8)rR9h1PLy<-FRYi)8;C|@ewndma9jw&ZBUu|U2co55j+{|1A5Hc-1J0Wnxs2SC)&~^Y-}G;<WH?wlk+rmy4Bj9v@ntS6tDq}y2}eQl%G+){UD9kw$>SX-HlEq@8^ux#&tM!=gV7KIARXDv>dB^IL$e<4?h0u@!iZhAGqbtzA0sGsM`afU30<jJLi}Fq^S+#7ObK7<U3P3S=l?=;UnfCT(_@^$(G=7I1-q`<zp=Gbj%l|@y<iLo0oB6@N7A2l<dON{af+u`{4Ar_$YmSHh{&uB+1o*hAPk<Aqp)F91@&iiv{oY#h*f@Uf*Gve;m1k0HV{ID~%Ua&O+tejimvrIl9jFZL8_T4VQdEt=76q$o+ERZ$xUaIy$wlUOO+Y&l$1Jg`rVf?VwH6m1v1qp<@3yW0XSlYy>+4JBnwLWo`bn?FODxnP}4iQTp_n{(W)p#H}QWkDhw?e2Q1E4C=to8G5vP%`CcWHZ$|8u2i{_)sIwowP)MW8uCxjt@wT)=ZQI;@3<&BVRo;Gjt;5g3?xw<!q9x&{f}h!A(C?Ss2nfS5b^DNL14T*Xul5nns3>~(Md>*CQ{W;0~XZ9qR>q_&Vk&sVV!eppo$mcIL(>X4ovXRwf}TK$3;*3HJANyIB-nPYua%2Ig{^_HtgbC9^>op>E(UiO^=1<Qdr;G@m8L{c`a}Vy|rf*PwyQ~5L{fm?!9olUv8}-RK1~t4w61E*sKg1?{6x3+^sV8M+%he786_t;V?wB3U|G=`DX5idI;Q7we!}>7VynQS5aSM7<UzA(8=qM>|80dyot^}X85mC+xW;v$byo;b55M|e6L3n6jEtTeHp#awqqbTX3TJK?9Fz0y<MF-h4Z+B(){nCF|Ls(8RLlEN7ZXFF`U<r(Tu;e`{y@wsKV#XU~V&)VDwuFQ3#@!v(NYWv`fvAn_<oIE@+~+01exN?qzTY$O-Y@^X~BSCR*34U8?qX!~A(1%$wG2NTGhheWOHkh7S9$W&Q<Ncu=lVYZ5&ZJ#UH9ed14;IS^qD><FpI^eLoH_JI2`qZ_{QVUNh6$7fX}2&qz|5OR<l?A0rBph~B~M7vM^Gm_mx@7Qcn7Zuu=H&HM@WmsG;cJ95q@rL;cl%;(ROV&};IDLSrZ}B1J2fHa5S5l2)D|%ow6hhNUt8KEpBX`{&ALO-8OEHj+2X{*AUyXTlA>Josv)A&blb(2qde?;G4(1oxrfw%}Zs1c)enZo$E4J4wR>dW*bd)3sC^ZdEX;KWAd2sN+i5}Wg8<Cl>@r(7>a3F5!^v7_d{L@BB77Td|E)d#Z>4h(lue#yAfT`!>YB&lLy$gJ669>R)q`(^lzGp(3qmyT+yLl4?+!8Dq@O-Nj^#WgNB8zw(_9<_6=ws#VBQU+c@_ZW&Z2gsXl9qk^bbb5MZ5W&W?V-q{8GYK+T|GVNXnIimA^NIs<im63?vz~Nn#FDecjJ@$MuG-1TcNc@7wP>}OTZDmzncfpyMnok_)FoP4>OBWFx|OU<XvfnV0{p<y*qRj`p^gy-xTb9EB=SL&3bSeo3oEu9Xr(iKGfHd_nc1qM-igk9YlvKi=j02FDJm)y}s#qXWc2BJkXV2{W;^m7sT#9G@N9-^IMk=tkT@PZBD1^c2+1`&&{~VRGA0Q1C>TQVv(Vh8~jl1prcl`RrPvD5xk$HYPDh>o=;x?vsjF4M|Q*=6;#6kX1;8ie0AScuhV?)@=}u3=E}39(~akX*>h2oS4&x;aUX?~cf0-Qk6Yk4GmB2$lb89-y3`n?^@JuPp63~_WooOR<@pU}X64IN<na_&uZ2}VezFbF6^3YqN{?#qwW{}s&rDr76Kjk**;uMD?{R}{q=r*Eb`Nc*y681zLKC5v{~E9E%t&Cb8Yvg5rQ*RHB9qG>BfITe5c1d)19BV4PaSe@5`6D$8;g6|T><Mx>EFKeg=p#aB~?lEKt|wjKcRP!f6!DEEEnHxf6?AZ<N)GGd;_X>u@sHIbv&<tV5(zHF8%wR>h_2#3@2Mz8{eMU27zybQEg0a)FrHG<!^|+J?Q)l6@OA9VWcMK)XZL`65alUY%G=OLg_(@q%1Jt<wo5%bLPT|)dg4No}D3}9^_$A<=fY?@hn=M@WF2CW96`#xzh3Ao#lo2$<oJc-TxV*hZnxn*m6F)DH`L=iG30E{>*y}w&giF*Du=5>94*RFF71ellB6^#eJ+DILRxnEBgHc>qduKKge>&H`v&+Ho>=~eh(a)CgDkSVO*PLX#O!r^cBaXw}Kl7IN^*EW3VdR6ub#3SKqy}JI7^OSL5+q1XRD38>!7kjqyR};_ty4f0}|Kei)n3F4gb<+@na2Y)&6P)?=#Md~v8F+@e%<e>RiyT{gWX6Q(FU1aGtWyfb+GQpYTKuw8M;Wr8?wx$`-5lcQE>HeKzgqLY{B)@lp&8?06>8`zPM%Rfb9hCY(habMs*mC1g4n`k(`m}`Ib5vd5t>20q7C}_?)r+PdspY!E-(tHVa1%lVsX*)71Y*2pz+G#K)x5=|&qmMg(?j29&uY|}4wnHQ(y0Bl^GqJRdd}JAjC*2EoMgTQftQAFgeb$HP(<=_l4=n<V_SOr7H%Q*|6@N4zH;Qk2O^AlV9dln|K7DwMJ~~*4c86WyQ2!gj-=sIVm^ByH#xRIJx-ZJ8b?lbO9@stDd6bsrW%exj?KB)#Q+0t6-A3QLe=y;UkXvM0*dS>g<&bOjbd~4j&eBz|ai!E23G?h3+V^=|<v&b+z?ko~9S6_(dbDou;`QhpZi?srvsB!znZ5Fj`KR7?pLVe=IskUtV)a>l@2k>;nZT|M2D-T^QQOX0E2kJmcL$^JJi4Jm%cvF$txk7({!o7$KfUz`mY>f3q^^fMT<7;eN$Y`*iLN$7J&pq^skD~x9PY!n7JKvYLRHDut;IH%5c3@|xxC$+x)ZS%j+AL`zvM&Al_a9H@@sUCC~&{i>HldE!o?Sb0d{ABPjMhTRuZwXmHi#Q2-j%Hx%BuYci8wd+iaJ??#0v#amjAXv|}b8I52KR2HVFufPs(~&}+bzg1d(3fjF#m<5ej3^jR1FwT{Nk+^EfXw)VZH&-$^{sEiyx0g5YlwEp64%j|_>r`K)}&(Yp4Qp=-N{8kBp8K@##(-%un<7jrDXh5{HaZ;Sy#t7^JYFxY{=Baua_5ApcdJQi#qkRpw3e0Lu`&+e>-EYZC#?b9X>`Fa}_rV!kfzx3=CxYUkbs)<$KPYj<OowWzcy}pJ*bQq_rO+QFE4dPz8>>^DNTqBu@bJftRH}d5{s;%FRSAlHo^DMB#rvT*Alw)@Bkj|GR?h+IJbj+-$3gcK3>t>xC@cq{cja<srQfQFIA-l&x}ld`@z~!9x6-`@?p<BU`i6)#UO=nLmzeMA#CJ^pN3><j<sPZ=^m(_|kOLN%92})q_`bh-t!5%0ZTrn?)XARP?oQbNsB_87>%;nT-s{{%YwF6;1A5~k=(on7_#{=En*~(t$%W%EU5a{TIGGIC<j+dt-SoU24al<A!jmcSeVNR3>TKJ=Hl{*9e&1P_BPvl_ez>=xJs}-;ut3OpS2LaP4dAz#*Itn1HLcEG`z^Q~Z?4m{YJa9<e=gaiwAp%pW@2`<NG^lh)bs*?Qkkv{_9l31=m_Oz2|3)tZEu5cr|NFw&ThNS#=E0Xj&4)-x{7iN4B+n(i(UQF$v2mI(f&<1v-v?PE5m(cv2T>?#J6}OkD?r?3I~~e@H&}R-=V_#O-)y*O10kGSnLsev}R=kzirH)-<Kxg3*@R;Pstq#!}>>UwnjxZ72oY)bLb-oewg(NY&Fk<$&;6UT`vt4OM7k<Pb}}zf6o<SPlZKilQe~iSYZp_%vwAbxZ!MmJ|D6@eZ=wU%@(-Mu;1<ulVU8Zbb9SH@v)urY7@vtcn%3wO(+Cx(2*yQh;6;UzOhmWO1yrdoBOgPj&|eTV1G@<+}-<BFJqf&>gNYcJR#^)UI}-!TA(x6tvy#%TvK-ket2+Qr*gYi*4?BYuhY3w6?eHiDga8kFRu^Y&y2s?Icx-@K4<sueOoAmCw&lRE}ooL)2L1^Kzg>_DZWT|rdNXkS>&+FPHhu^*VEy2np=Bt^jfwKH=o+nk>xWQ$V&02CcyO7c$PHMAw=zO<IeJeT_26;(%5v(PENexJ-*vLAk$t(>yfzkUqj{63A#s&+>g(9-AYG?g*?n;`!?sBlQBDddB6jBBGiJu_sgwnr<VL=LLaAJKbTAU<T(;_1g+tzaD>RRh)#NXIhra)br0*U@*-b9%lvYgExpV7{OOV(QEKQ)z9;j28Yj-~iv?@DA*72HreGVH6gFNWH|VIm=Kc*a=_hOFs38oL)@;03)q@h1H7|q@_vcUPqbwId1za`nFZ6n6=*H?wwpbs4T8N)|rGz^QD-uKM{t5^1El>bA(^r0EkJ@wY%RtsF2yiYFe*)oT&^xY{>ZEq>b}BQa=^3xFC^6^xXE?E^EB!ngGT`;t3xI9`+oM3Cx?Hq|VMtn|i;`!>+IzOaP`l?nn(fx=F82y;fu|lV5-NlX?Z+LO;bM4vT$5J+9)uY^b?2S!(^7WTsNYFjr`fB*4;}^(Q?0|*ZV!8!WnxM5r_y|^>6vc;)ST~rrp+q~5fL%I43sG{1TGUd#cxf7E|gb^T@Bj(QVAG0l{R|NwoVE^<;kSVnMZVmr_b*X+vv^GUW*i*b&SfoEibW$-z3j}QhAElPT=y<OESz*%ms|@E*_-21zWj(ik<R#;MC3W2&#75qq$2?*|hHtk52uF-;;P-a261zyDqeyXYoc@KHC*eXkF4^!dEE(0Uu#)%MAaFfsq#I*o$*<QZKq7W^DY!CHyCBJ_YC9tI73}-UIS|GSfk)gf+2Or=>VrSjQ`+aWb#>y9}RyoCv{~PqDF0_LDr{HRhzCO!zPiPM?aRtpC(K0(eaMI6SZADa(`!8Pv-zj}Tq&#;<x?0$ukB?mKg5jyAHPKOHIaY}U&>X}5J}5^+1cy5NX;zDl{U=-!K#+t^h5-CMyr2bUm^Thwpvh2Fw0_eNhtsYOT&zY3kgQr;e*XZ@&0jc=I9&~!x(qB-TyY;JSnTW~+RgU0^U^xzn-C#Ib_tMsUxi{Nhr%x=dRduQUsgP7!_CAjSxvQzKX2LLXQq-po~c!w~g!N;-C$Q8549h;C*aW)I?5M+YeSxeMlT}&{YC5WP0gb9(t<QY$Q%9hXK_Mto%{yaFBQE7|!Ki5?|0)YkHrH0-~T+zb$v9B&G<7F+J7x!SJ$AHYT>y;UOSMNS~=nhkjvs7_}V2eW?D;=T^x<09T2sKBs1;CR!VMybvzdr!|zgBHdi((eC_ZUJemHW~he6Aa4CS9`G`Gu-@6`5e&TJNJ4q}^b%qj#(XP02b4z}=MmKccQ{Sy^TaJ`06V0`*myFqD8838JDX>4G4Lm=F*|Fg*M5zi^)y*jr#`R`;6jUJ*YTs28#5RTLMfL1ijCHcSlsXTp&8r{xeZQ&G8BgAvhDR$zCD3w(>ar`%|3heo^|A80F?t@3%z>Z~7lb9IrQvnu?|M<s|ncbPe|9MJ%ZP&4bVH`00N!~15ue)7`rzJpd^yBK`AdpdIg*dSLE4n`lx>kb7`-9X59jr4k-liiD=W9ID)?>KvUCCHfy_l_TWIaUW3gJw{EO}^_}Q+D}ZcJZofY3Xsi0-y1y6oD3bwlSXNqZZ3bVo<b)^*&0lMltEu#Vp)7??{Cy{7`>R;M<>p9XJ#-fd66(%%hbx8+<DXy0{tTTS-U8-gY(W_0OT`uh_$^%df|{tvz-e!<3>*LcNz(u8O;eU0RdS*lSF4G_CQ@^tzrNjI@muW3aNted8xAzt1muS+KBw^8<?&9-UKaq7?p?{~V`CzV0*Yb^`I+O<PyR<#Xpgi?qk_-f1eKKo>h`BXocDBfHF4!uf-4whQ}2fz<It51Kc}{<|O{N;_C$6L>qJhW6W^&z{=kUnhe8T{Wws54B}%Vg+i=ISki3<9gBKCqy<adW7;0q%t81jCpV9!{#X927Hz+SADBS{HejRBi9Z9ynib#kn_xN@XlE7U|z!!R*lehO4N$2Uq0tyW&?}8UapTPRQ^S&Xy+_mjejh+e>EGU8nrt2jOU>IhlO@uFgXe+lz*1ywU3|6LePG^Z&dBCj2Cs0uIYEc6KumgJ=Sn*)YOBrDc{$tQKM?nt89W^1|2-FV@)|;N{uB=wpy2-e7l|${lTz8l<oWRN{h$6^7T<oDt@rI>%{H!xK5$z!3}<#DX3q?;m#lC3$G?F#PVMc+83(V&a{QR2FB+ZjXPEBYLfN-o!Aw`>DXwO*R_L!W=^57eO}sMlhRg8rebBFSYUCB_S#TwH(GqNDc2V7&qaSOl<k((sfgZf*m{l0di~#O5jUHA?NKb5q}B-!Ty33omRq(*pTY7W(&ElL!hmrl{_-X0J~2}0#fqj6cgig&ZSLgb`s3WK>x>I7{|`+`#<ki>(ONGJWZf?uEuA12-=b$s?*yG)Ljy$V_6&Wr9M_&$SSJd*^j*D_@sKA{t4E%J0&PDhjmXauy(A5Km!T%4fA+xF>3D0x7RgndfwpOs2M#lJyrb8V+eKwW3Cy^`;}c4?m%(B#tn#$cXz;XG$+@Q_*?oMUj@m5`r@-2xeA#7t!VwznsCAS_z&m?$WaqE^7Pm?j*G>zV)ird}hv{{M>Lt3puaO`9uv|v-42~zg=VoJe$(%fG@6jyahse=XlarNJN`ej^@Bd={uX>o`b^m@n?ne?2WtCbEk=+Gn+lPIl!7(8O2Gu6PC?k1seJn=pN6k0+IoI{b@yRV=jr#fBX$5L=dVMMMN&S?)$Oo_IC>h?#YhyNwe0C`QJUGiF$O_c-rv9a7$mM10a(z^`XY<_5mR_=%a>aHt;hw=hMx^Rue%mJd44Oqi{5aKfS~Xr~mI+uo1{PkTZ8y2N7zNenwW%08HiGY@5-`6G9+bE4+gsCBh41^}Rn_<&Joh2VCJzkwHxL62HhaQzG^}PhYznPEDc*E(la$gSC4@??d5<xh7<#(wJr>0y0n&!XNpNFA&G}u_pX>fg2c^k2nImcoEf&j1;G);wY_vN`6z1MKrg}zvY#eOI#@5lVjZ-l4kB?u%1i6Me$oA4?AV*hGo9`6UTyTtSE{XNW5Pkj5*gJk>p37AE=7d5mJ?LvZuESk=ZrxWPxX$WVbKu+kqbzrPdntUx_){Mf7ku9iOd2<@zqg?T{B!~;VYqx=CKwACe^RXFXN}Sa$1d^p&*i8qY-6{oCoPaA$^fA&;?<gu2gG@K9a%kcaDIGN<WA#0SCtZ?nr+wh8_8!YIS;i<;dvFPabL}DNA@Empw@q_i^|;lk>7BnISI!s7aY5YJ->58jn*35IK6qM<TJfge;vXBba?d^_3CDLqIpmc?0<jidN5tlsPf<r#(l+-v8gt>-WYBb+{9#>UAn42fR1#>5<uHc#9-JLkL?P+sM`l-bX|wTF}2E(n>xH-h)EOyWYf53DfTbB=KofQI@dc)@!{`nT?Ti7?XL27x(nioq2*ruJ-QcL^4e|6?pQ&-C`0M)QH!HlA|21VYkpit2beeZh^cpNwI?dza$b3%c9qKQu^KX~tD`h1fB<j*a^$)HGr(1^S#!<>)*fZ*^sUE_oyk%yC2aS+#P_$hC&WB0RP~U3+n)ZIgiAG2g!58a*h2sW+`tsoo;lAq{YpnY%l4;)+7KJ|#KW_Z?GVRcaOSu9V&_9Xt`)*sJzhoQcALkmV?pT&m9+{;s*|s~8JMU-v6@DFTv}gUNNhr!#voPmd41m=72Zng>*qQ!=r>WZ#K<Xm5m}4uU1=L_;w63Zqs|<<z|b^Mo)~NI+m%thQCc%-pL+S9=DF=0I<tG>tK7Ci=4x51?2(*D`Xl#`N>cFYvU5o4Q5pDqcV}cyj=tQz9`)BcMkRRA<;{xT>rTD#qrPQ`2`EoT=gRal?uzADsQ1Nvb<-YgKAnW$i;LIZk(LNOS1P7s#FtDfy4$}v1(_Qk)Ya=wJga;x9`kWRwj~V_YZ}~k&v%R^U*T|cTt!^6^VYrfXkN;hgH!iNZr__h*076%n}|HNx#M5s_5t46PpJlNJEn8CTy_PNl7@U5AEH#=OggdU0>>59T@Csh+H7uc=nK-H96)(fj5xkxpgMf+n+~DOYkn=eFs7CwuHX#9^?Fl#s--zoSALmtli8Tb^HJ@EY{4OaEa<#e&Vc?SKg}8p8|_OKb?)}J&sb|C<C>!r^KM9n%;Q=|sPgFm&ph>|5l83RF`{}|M}~i1kq)9ShtQ)k@f9;eaHoWAoqCS63s>0F4;vig0klz#%R|3WK-xb<LuiPOsKqT=EQ)zX--#~X*L=r)F0Zs@s+)BXI2m){jaN^A4&wv+m;K0pi47sl9(26PVwf#6tG`PhIR@XSTdqY1j;=u}H%r`JI7fmv0%z*rzHSs+Mo2x@faPc;yGFe-e)CwR=e6Ijai{YjV(JJox@6&BUb>%Q<<ttuMu=9I2OR<VSt1XA9kVYISEf^!WxD6eUZ<3xW4C8mt^VohB6`u!n)>$q=-M`J<KeZ?A$1#3+p{<%$Cpw4TIPLCgW+!LQN-_^s6)v6fZ|nzD<Tr#Rm;!*d+Haz#&k=<c9Ua2L^NyC?QX6vMED&Bky^_qVX>cmnBIW$Q#l;h>w{fn>s2urZ1O>(aM%CzvPf-Pu1{B;?Ym6sY8`2u(i7ck(<$5JH|S(t*u?J&Rc-*H50wN3Yb7M$^#Yv`VIreeT*s}1`UI{$U0?S)f=Y7B<LQbv^_hTkhY^tB!@CF?y|tkyO$CXxZxtz)G!pM<YIqv;Qk1RsZ|6$!yI?_0sVKwU(c*5apnK1^r`Sa4arOG_)VA*v-=$~BXRb1-aWVtF+gNf+Q`<?ZupO3Cfc<i2&c0-+=-sNZX`RJzv>x$?%UCeS<yPiA=U*`YTb#>=HW8jYxkBT9P9Phynkju=?+em%B~`8$vfmWfDX<z%GH4p={|0Ty=z4{Sw}ahmon4Xs047dwLdT$`?6a(5)+fd4pQJ0#?(<d0_2}0cSR9a9&du~FSUuatXb#`u0;*@ntu++I7yZmr)%omL7<0bJ+3DwkvYn4hQu8F-$9i&>f|MMk$gOTN*48HBFuyRT`%YWsIE{HV^K4yLKTG_&Zf<H9c%seh3om@CbO&#}_?2ma#&(MLko75-&#kgk!acV4dY;@H=v7^>KPyw8j`m1i*ZR7El0P37?XasUFH@vdBO4nx^?U2bto~|8k5wsGzqQ%8*W@T_58gF}8IIhRtv+cL?#m5^mr{6Fep>Xd7!9P>e!B(#lF%0dLL~lYzAV4J>VYhm;0A7+qI=w}C+KkTS~Ys_TVpg+xl8kT5NIf$?(jo~L)=y$cY)Vl*n<T0nB1_NG9UZ9(k0hc<z6z*K&2gXL}oKTNwia9f(d$r%{@@!&!VU!S2F##eYHocH*;&9YUqf6aVXaJq!ZRWR3{61Sz1`<9nVrmC7qvmqS26sbL_J_5qT8D1>Y#`7?m%3t0)j!W_AB(T3hHl`{=;NKV4n4by{OP$804X*c&zi&O^GfxXc?Z25_eGh}S0Gd?ibd9WCBVcv0;X?i5TPeDmbvB+E37U2bP<6#Y|=r0!O0M=Oijt2c4BI}7C<1v<qoyZIX1r3>$!zlU}^A7mr%c>vMBaD95t+*?ToTjd;+)oMv3X9BlhtCCS}kK&tnaIzy-E_;Ol$wT_+ifLTW2agFiLn|kk6yrp0sGjJm=Mg8?sW;Q4%JjwdGv2xw#rL58H~XQ)Wo0!#m(<O@1UC_6gmlx{VpGZbda)vpbwbg~w=^%ifmj<suHdWnegmcFCo(+zH@a{bLDk#yHb0Cx=u4B%)#hFUrTSy|)GDYDT34~cCcl98KaE-KXfyq6>WgEyjJ6E>R{EAmOa_OO3`rTfW>wmNbaw{|B70A~8CQ2F<$4<h#wjlhYgJs&!mWr$#Uy;M&$d31u4B1&nz6FL8w#z#r9<a8#nTl0#1SjZ-pBcNGsw@a^BTSZtz9_6lwMKwYpz}sDH5H`8k-yl?`F>l>fUfTd<F-A54!X05Y!6U+qV}S-m>XI3z)7NcJr-2C6ps^_5NdN^+Bif8X!s4zV(*=WwQq+<Lqpdxn+9R)&$*cn3K`DmEFDCYy;=x*`2~->UuD1r^;c8vD!JaCYgUO>AgM^!Cw32-`a=E_WP6R-5$T+r^GLHlVD&|#!3{_AO)<}HS%_P7l-ShS45LzX$Lkpi}ZHqrhmaSp9<2Wyav&}G*Bl@HdVtYeFlq1{}@X%ge#@%UE3`+N$fd}@&0#PRabD)VUx7kyw<KvvV1z2^*u#c6v<TOgcSGw|8L+>KUJmYv`pQI9mN;B>!?^ln%QngTT5Gg@GFo`?XY8CCuapLEu5ytRp9>H>v*v!m#p@mrED-wPj}7ftf;KQ=CWQ_Sp9nIUY%gv853yzoK_EY@%ctxbG+!8w-WF6!G$tbwX51Hc5krzP)01HpFCHE(&=E5jW)C$-1=vZiibJr<J1rCPdi!Vj$GgA(t0Ld{;vY>s?&d-O3oLtlkY0@N10)aGno*md;M(Fpq^%=)&aH{=<gm=w=y_!|A6}JG#nP<$nST95EzYma_dh|0Sfc#T)_0lIFn=PeDvHii+CCo6!A`gsu`JIGyBB#Wi4LY`x_t=>2i_u7NXyMc#TjyO_5qrwL2$?>FzCIOm89z=&gkugs#Hrcr_<~$^(g9hf{P3M+$YaSqL@x``RB#3S)iXe<$W&WV5JrUeb@!#md@FsWMxiw;DXc9i?DC?aFFffbvk^5AZ*i5;V+4?R$5!*gc~A#E_ecwGtx-&ik7Wrx<(VF*7<HwDc!^Bs#%mF=!V1X^c{cfah#=)x46c#V3W!prE6XDZ5Gs=EHJe>4{?sb$@Zs4(BjR_T2{^POW~?3vwf*jFOsU#`BD|xAVYp7hU@S#fZKA2?}w&Z0rhTdzb4ml+2>ZKdAo({(|Q@!C%Q$tn(P|jaN|DV_a>J+JbHtG-A;5bs-bGaVIn}Qjo=B+nYqUY_&`WN*f~4Zl4@fiI)~Dg&>bS=-ge^A6-Q3=&PCp-DW}Gu>$OOc-|gh4iAA^zsJUI*<zL}4>|zf#jU*CkIO?#yB?>Ox+m)69gXvfrh3+V5XoUfvyb4!o(*lc!cHv*$3CWR&s&|zr426z)NI@{#Vjw$Xzv&AI(~Y)HZQn$a4y}%J9l!7wRTivD@&hk_Kk8*{3dj43wa-Ypi7d+w3a&O|1)@Ukps=s+GPV=!XP*<r=wxx^H<nASI?c@YH8T~9#FK!bLX9Y8W}Jgoyzybq5&<Dp=xxlSAZ&GO(4|GXHaRq-<oct`pApKrXbE=t(~o7Y5k&PB>FnZ9<5RQ{|HaHEA>jHFcbS!Rp@Bc$nP!SY;_p~Pf{z47KfQIvx8b5mjLS<>4i`f)opQHJ0P+@B-m|H-z_SyP2sXlxhvz1Ly>2);Q#BWdqrN`*}v)icLhGaRgntZ(4POyXtUWL%R4MV{|wZ+CRvBmjy)yMYStk1Z@If9%sbbTWpB_Z&*nZi^W=fPz<MuZryL{|I{<f*4(1yi!^=FSSuVEis_iSwT?w`;K5ONQ3Hkb3#e|byUMBQ!R<+}G9?mn=gzlwn4Ug7Qm1dSRagl9b2HY{RM~&wsIs?~6y#y&sU9H`BQl7?K>7yJ+7@v*sm40<6+;z5oE=AavD)+Uw4cVDpAC5NOVrTsIPG9g0b+RHnU|6d+Srm;@Rq&@9y-$p~`-yx?Y5Af){#_PegG#~cYgn?n!Ufya6lB0GTDDbaG+#Dbqdh=C$|xLw?cqQ-;x>dSSjrqL#c#>?n}_S9U3q3iVIL)`yqzq}nOW^K83?5F$wA}kKQ@U&@$kN_h3^^bK5XFgoMoFnSM448)Ob->2(B_PVK)7|p(4cEkG#5W8@-JM`-|3dQ+|Z0B#X^t*O^Wq8};BQFWmvofqRM3wrt)b5UP<sC5c>ebYIu0@0wu<n{)7Mq<khxXW&fMeeeEc_ujp()sBz7QWlwTHYNNWFBB)mVsG(Gy(Sw&i?`*Fh*RiW6yU*g1`LN|c}1Mc6^Ka4=kW*FdktLU|4Ayvt_jrNJh{0A@m+&`Zb#%W;7e2oxUP@KmVUgsrN<xNt$(t?>DsLH7N!Cd!2>mHJxDUY!fDqPF=O@lW+@NulG~7O$+?*hP<-@$kr@+Z@d$seAze5sYWv-Aqc05sR-0v3h2Hz5a%)%NT}2u2>Mx7PXmrJ~Vl_|db2H*<TFJKa>gu>dUv<;3KZ5rQ<4rby3}rvG-8wn8pSU)q%10?&FnzImnKUf7qp<xB7%M}bxs3^44`+4Sv+zLK*O`m(3yNnaFj)o4i)s7}@)I5F*BFLdqx%sfXTvKmyPcKRy1c0eQ1a}@coW73vS|Z;xB$LCr7A2F1{WXUQ)u<N0UlCTe>S*#Fn_Fk?1QzIFRjh*6JL(JB@OIA1qOvQ>No)ZS*Z8(+rt(|e0q2tXV#%IIxo3u%`Pu|yOnN-Xm?kB^%-%4S*019vQ*r^!|e3j%^YjjgigEdI`sYdv+1Qx&Yu3V56ap$FK#o5+Uy#Z89-URzG>lGxWUw;sc1p*0kTc+xqp6-DW#gS1_6;w4F08v-AQ6ow0IU^SKJ+1wFB~T9wLPD5n72#pFr0vDqzw%)7@Kx*RJI!T11HdMKFS5lhLe<CIG+dKOW?K8KDnM-Oa)YurHsw=B?X6qg8Owf3k?@rrXGu6(w&TR?}>UM18XJO8u~QpM+<>cbMao`)jO*xN>;EBQbb!g9jVg>7-gi+pF$zLiAmnXL&Hh-U;bFZDVRzDj8v@nXx}OqS#n&e$vG1EV%>k)tuosMa4e70U5-$u%8@_V82zK*IM(G3~wfs+=pBhl;FC4cLpmV<zn1x1J*LzU6TIR>iq?6ykiOw`*^bs)ls`rQ{?4wQ3l-H-n9BNR6I3yx0{%eV=0`^TETHmJmH?`w|AZ3X!-YUr%p+K_F(c-eoc^`YlS#Z^fyP)Hl(&%4x3O_Mw9k-FN~#4IQ233vii{4lvW)ftg5?z0Nwhx&xgvX#6B}&Rdb8@;_r!Mq5j<@4R8=~>yx<v6%A?4o#ab@OKwVL@+@s?6r3jyX?57tnr}ND-yZL(E(hu>*&Qq6M|KiQ$$h@NjqMY<Q&v&FBVwvfWOS>gZ>qqWb3S(@1%8A@q?>2aPxsl0d8_H&QF!hj^f4&}SJZT!?3>rcf}5qDx-PyfC`#B)Jemy~XGjPrkJ@eO&<w_r=g<K5p#uQ+_mM+rrZDts{7=6kQw7@p=4bO|z4|3M(k8!&d)P_p!<hED$LZFfr`;A{j?02j^H{&8v=ck(tH>8TLdfJb^df(Czm$3>`kI)ASUZr&3|>w_@Uia<O0%vm3q6$!4&D5xlnk6P^Z(CIwD$};mh^i!-+yFSF0d*%=b=gIiF)|!8tZZ~HLF6IUh5<hM2bqcc8WXxxUC6jvKBbMT85Z2%&EwZ;HKQC*X_@!4a%Rhi#HL;e+)+a?_UMbmuNe>U@ma@a7>}?0QzuH3GEy8l>rfVX)0l}0qkhdN{&iM0rfinHo~n#MF4MnnezOJ{h6N2usNw~_IfAROcOOP!R5=kdv@A!ru{4*T#t*<J<rTy0f+ii?{dIs#-ph<P1uDCwx{n(&!%TfOjF3bTN<O<9+}N$W|_7>O}T}QqQb*}dyT{In3@Wa(I=*(?07dCjeG?CLtpj;(1kWs7!8%Dm|01p>eW9iSNRq<e%0Q9T>(+d`mrm45z<chhW)H>4auNqNv-^@8MpSc2w*Hk^`KEcn)gp`e6j9c_ilk!f}J7rU>>->0n6zi$XnLXDR!MIeVPVc>equFTkihG8b&AAX=V=WzDPD6`d9~Vjo+kqTvoBuoP{*iwATgdPH%JIGsJSVIWDC!U$GO9JM-YG?a^Nxsot%pvu<_GMy+`7+Srh?EmO09+E=#uV)5Z&aW8h{=i-35P|cEE^=OwhI&&u$O9`3Bo`+;U>CGzzaKgdGfVj+IPMOX=hsosKJr4-T9AmQ?j9sLt0*fskR+%2NE%XxxmY83rb8KV?5O=oOd~hdKVUkI_JZ1URd)HuFvCp2`YH;9-uWIDd+y1+n%#@te7K;^^ZLiGC8)Bp{CI}W<RJo(i&ZUua!O91Ioyy@9uEk<K`&k9=nb2-zgtXe0PHma4lr-7}xP0^MZyLY}@#C@>;rsKwxJCT3RIi2R5jp*Jh%V<?e^4sHa%sRh<80cp9wqR(OPq7?OqW}p-t9u6;$4p=o1S-pI}j@p*Xb$)F>5I52dGh*wh{xRdi4l>P92~ac8Mwf=eyZS$Kiv(M{IrTwT=&^RgcQwZPqP`YX`!_)ON6RrS2{t6$8jb?fPPE`4`e-*v%!AJlppS+<w1P=8QILP_ss}Vnnwf0cwr|L^o!~n5VlZ<%-)^n5n6ZUSdl%Q>WLMb(}qS;81BTdJ)`}1`9~95Y-ZE)}nEmhXbeai}_42z>WI<W@GH=TkYG8{)8BSe4!YOR8q)1j&z+2TdAO~ZjWFGQ&$9~hmF5PkIp8HoO64^ntpq_JpzYXC0A<e>TXMdT<^}kx;1kPAA8Hl0;DV9A<`tS{~5{-lwYz>Cj!2`n}x&FlYuhb#R?rJoSYJC@iF#IW4S=)a|k_74ji~k@&$nCF6FJO(Q=-(_8kiHMpd1<H9cVy)B1GzI?e|NVGaNr`%V$F-w@Y^R@#?l?w(m$%KKJrpGr@qe`P%_-wA$a>C_`>vJz;u86A1C$R4xWef>UtHx!%f3<`;%?&8pSzriVgu@e0CmzWL)C{umxpWm(EZix@)u6l=d#fqe0Iuo}p3@*z3{?SP;3q|(TfB$@@6cy(g=<U}Et@8LPtYyrhGm<;eR;;a`@GY^*_kIZW!NnY0zWDGp{;5fV`&5#~gkY0;pl${NM~nWMVJ{ZRoy~g6OyT-=*Zk-9>ChwQ)DR&nElqm^Piw{ho9uV0<DtRp)OG7QFPEXyqZdVPmA%eBui+Ki@7wzC;`7F9%gT9(AI%+hyYT^4>m%!U*R4U?rMB(qz+oi_QLI<HH(=&<yK&Pp%bOjv{l_r$M}J2KgdHySkyHDfb;UmSu|h3&wfHfpfq(bthL=lrZL^jIrZ?Cy<q3~2p{Jl!tZxOFA2srD-Ph`mz3~tDZ)68oy__Xc7RY}4hBb|H$(JGhz$N?9?oJ<8v>Y2wrau{dWUBwf?-EdQ<3}df@3z!?dQ0<kNww)Zg5z2+^w#yQyW0BQ<L15YwL9=rc4gkzHA;$(&WhcpYUunorA4De2e~}*S#j9V&9bAk!6~|#yr*&7b*2Civ;eC0D?MtK8EZVro{V=UTaRh`Xz0gL*}=B<_E7CcN$a3JivN0oVhQ}R^lqsV&ikHmW9j@pF0gt4xlnpz_p^A!*bVOt-W~?Ed(v@`H4)rDbjH8HAN{`aPMe*_*52lz0+(SZ*O^<-ak8pQ+`SUz`UF|6l$_7LuCM0izo8<`ZDhczbgem_l|RFRJbeqPbQ)ZQE^WjNP;;tB5tH*1!#U+DME#mZ>-Fs)Cfz#C2CVahli>eO{O6w&@TcvhU2o)%e<xyw0vI+MWfy!QW8a1TPg=gwddI!Ugr%oFup_};Hjdz;L>(s<WAP5}b-@<9uS6A#*!~*9`1arIJa&;6=lp{y_~A?*opH!5Uv=^H#}GhQnW{FWCjhj$A_kZ(@5E%8Z2I6$NqYN7XewH6%~l^hS$H<%IZHC!Y+GnX0^I3%p4WVBDA_mX1$cai`SXBw$Gjfght9>ljmw=U+8d8FXA_m?`IA*KzH;0&gXgXun!<N4*YjFo{0W*qcX3>P|7QR@$tf?N_G|Yz;<Sj=%#t8Z_!0Fo^0Ge$oN*m{<(oxzoEpaT)8kIn6??@G*Yo<lw~X$PrFC8uSQwcJrG;C}Nf*m*y_?qhjjJ4MBv~teHy2&wuFq+*O1O_XxKE45Uq7)54=%f_yJRK2BM@{)2Z$35g?v6N56x=*Vy;j4zP6r7ub#PGl)3geJ8L&H_K9?dDsZDlH((5JWuI2Zbf#f9YD=O^-geQ&_3l{jSXwoIb~H#CY!Ji^I^7R`+)UN1atAxhK|}SMC%xHO^S$tW9$G4Lu;}T#^Inm5)}YG+g|S>%viB<qgp{S-_t%^FWTCKIGP~v7NxN!)J;~E-qGqSt@^LFM8<(C-)x!qZLnWoGxG7ej4$5n$ky^`RVi_E(z0M&<RJY4uto2K5FuuOWlf$@I7kI@7wBj!wZlS>hO+1I!vA><){L0FT`K`o<{w6&fqe0Yw3t4J?o}KODuLFjG(@%e-zg}jS&fn+()5Z<|`Iy&)hHRILXYneu^_X}c@?e@BIz*cb$tQCIo5}he1&+I{|7!sMObSo(R~Ov!P}kf#*V{;#&g?ez7s>Cqi_tA&L<9m#wpip(JWMRQ*?1ab=Rwpux%XiAw1uUoQl}U;p3gRA!Sh>kBxUHl=s^_p-46|oHo&Leem+p1L_4P@pST;km6Ctj>gD_*A}>G7pAl`L;&uQs%g`LV$Bx(~^AFB%21C<uck^4$=?zuM=Tl{{3VOo6#LeZ6G+Q)>K3rO<;N!%Cw#l)J^u>?Uw_Pi+yA?-o`e)h%AQ3P(Ra@APeFEklh}^i=7nfmg3yT@B9bbYZSx!HWqHvo-xqtOqD~g{S{m0U0Z^|z12Vv}OYt%5UX(D#J^jk(v8Yk_%a6`U#8kRrC?+C89qQmg~DlafBq*GuzT;eDH?R(eaPI_eHWsO5BxYf4JaN*ON&MkDG=K<q_?_I}^O2kaDxle`hPKT=>Oe?!<yPWPG&2&mSmG4%`{;bJxE0MX+Lpff$2Y#k+=f*SamdHE6m4-!&c2sP<r@A~+cTe@(YAGFOi&e{P&u7-*Ml7nJilgmd1)etp3{T4Mx7mL_E?4s+q6r67B_9+6|93}G6H4EMaCT(={`=FL3+EA0<(lUb`KGXZ@j5{Dw3+9Mzq2WflCzvdYTLuLyO-cHHq&%s-*G0#=MlULeBfU(0)Ox$e5Aq8ChFuDq8$fe11NUSFdg&U{+;}VlTMO8{%n^geWk<PsylPFZnm(e)Hh4bVl*&#-oh$+tb`_4-Fp8Fs3LZrq*VFaY8g>A*PAu)^e*%8f%Y1;>M<BA?Y7U_hIk$>qC~B`!)OJGrADen)58r$2De2Dc8}*VIypQWz(Xzv+5o(vh)r(wVm_d#dzChhK<!h1B&T2cSS1ze#Td4C>{&p@6^}2j^~M+Y^;^|%_3O*E6EZ<OM85h+GeyC2{rm8%9zGJs{A>7cN~oO{6sLXW00*y`6Dt2$^(1WY@NwObN?hm5xV#@P|D1uEKU>7exQ^CG*K%)N@b2S)i4@p1>H{f1TwkMUz*wz^n;&Y;%VjZnAh1O3msB}X2J)g5d1|G=T{4f5FSDhZR6eI^4frxvWppDX`FG0R8N5ErEpj-#9Cwb@#rCYhSMnp?Bzyf}$WT4IgM<JS9xTde)-#eVDW*Sm9gI)n>&zW~Xk~%ZlsBB4*O17Rv+(S=9U!Weh^&3EuxR#6*4L|e^}27A0&@N*C43dsz(5+K=s=2KGBY>F#5`!AypCt9d^%rL)mJO<gpO>;0&qKjsBC4fQ9RuW`$~A7!o9pD$6r)<U6v~kpr{7`t6SsxJO#Bbj-k^;?EMy7KbDG*-MHSO3iPaI9q9rBlC5Q>on!=AYarbNuc<z7O{9u6ps2O<9K5wWgz;sYIUToh*X~9(e;mi#$Ne(f4!YWfJPb+A&}gc4`+uI5bl3Q``^5>X;c*tVJ%`?^r{iSxGU|1CBLiLlZpQ;mt$B~QBc)sJ3<xKtM9_4Xm-oQ?OGF6HMhC}w`bkIJJ(J#{=!0AeAG(B1?)rbJvv{Y%gw3@WzH=M7%l#n7qwf*W*-Ut(n2pY4Ck;Mp@>mYZgFVeVdNy*KciLgXS{!~M`f8sF%KFkQ2bIHl7<zlbskCZ(xmO+x#L1Fu$iG=z-zwA}na9}QJJrS##ly7oPfvEI0Gw_PX#}(FPTtNsA|Q5DX|}%bv;9O>i7f`S@--j3aqUnQ*lW<Y^jRemmI`zJx6li#R>o-e-ox0*=9gfw?2N(Ev#OK|UZ==|6l!y<RZy1rY0wZip{~=rz5?{m;&2GFuz1@v0@M?mF<+Q^gI3Ab<!)3m|D%#<ZF=!r_VSXLN4-<hN7F`X<iqDH+Ftr)?yU67=jVs6bjXa@^O^rG92kY#wNJPFrjbeu$6lhyllKVidVe?(0a{77CfGMbdK$!gomj={OJ$$P24!Dph(I^>uDT2Q(NQVP$yP6fMJCBnJn=bnNybobOuDTHh!?v-jCGdf5O_Mo>s!V>bz2s8HC`V5Vz}1mDo4f3`959i$(R`RmzSd{cMkB;``bj*;o-r}2UBuLa&t?Asx93*Cqn&r$%ccK<{^S34;iN}X$fEnX0zv{jR)j6*OK3?&gtqdK=*@QL5%EC$AMs}Ita$IQQmUZb%`~)ZR+_G0bM_wqW0(I5~e0doYV$OznH$tkxAdF40Z1J2x<VInjF0ZKE>aC@`P5;i(P`qf6P|BFPm4C&Q3ziAb2{Ji=eGU<GQA!r}=Wn5Q8LdeDW^@>C4Xut*hvuavKA?ml3t^Qu!^I8MVAsIxg72Hd0pI`_o&vq}cEX&d^j|ZC>pkp#3*};ZreKp=q{>$#Q>1Xa~;H!qJEuCj%UQxmU5hNxG4Ne33J1w#H^KrU=LBPR^ZA@1UoR!Ta$t%`$s*!xk2I)&kw$)?)L$y}04a-h1RvxzH~G;x6C_tY=D}tZfkdEMH(TMyW|-t4{YK(;r>Me!G8OEI*5r1y8nldzgn`hT<LOPH5$u>JYT2Q?cZqrs=f0>dH7)J!yC;^DeJ4mj8Nu)G-y;L}vxd)?{xShN-E~!r5oHI^TfPsNT|Q_$66C4>di2pj|(F2<b)f&xG*a$;xXaDOcuUl?La+Q}g@Lt#7WucyE&s7S_#A_gZSamcO=lLQh-C=eb|ryP@>zyR53ldfXAj&14&*)l>adZrW666Xh0ZBvmmklZ3I{1-WXWient6^LS=VBBar+S$3~kJ1^N?X`q;uQ}aZqbGp(%{^lXdWyPa|N?185=xS>9nhE?*%Dtwwu4(*uNf1HjuD!kEb=7DL#>KydtXX-ev3#zgjwE)6X`yjw_gJ37)VvMSb<jP2KayWY8n2kUN?7z`?SZ`TUp!0?10Djqb9*3MUX#g9=l#oL%L8%tx<~g{Qr~yc1??J2;|p$ZX`RHNirD9l(0Yk|>I}ss8m4XUcJ95wMrIzm_`MuWie_{5nf}Tv`q*|m8K(92tl8gI^{PnyRSMOuTV&ng*PTe~E=9wjZ|w)l$}P~ljC;o}Ga6*gar&j7uXurnB=45(3ASskm*K)t-8*tCIhAHE3iCO^Oy(VVExU;%tg*)m$d<UpoPIeEy*hmR*ZbgywznE}*3k-e2Cja2A1rC9>C<5`)DGLRS9xBul55ss^cKCmM(PTP@F(AJ7xMYuyF#S=U>19Y(T>S##Pw|s?%<4l6N!}><3wqe51i}~Ih7;-JPyHU8SVTD@A7^jrsQpX4c4V1t@-_U`xq^_t=f4{n(4RTLohhctXpls9TKr}i@wl#9?^~Qp-}F8;k|s4-*$-W{Nl}DuW$9dw|akW_#EmExm+n%@SLWY>O87br*&Bx7Z!?I_2|ZQ*8aI9V9O&>Wm4#2FghN%!iBM`i`s-dHPrU4^ZhxSy)l%=Fpr&XW7M1~CLPYi+fpM9;TL1-&Yv>I?m<YlW2&G~;osSZCy}|EW9@cvpl7f1%seSQPP-j*=K6QR(y8<(_&}KIlLTqIhj~9*>Dn>q?-%wa{xXZ$MypwW+?ZtnBP0$fHWN?Fmm}%a%0lBolE~w%ge5(CY8`b5oc>8lV)r=$(1E9vod3Ge#=LgiFgJvytVe`$OKMwJZ}#O<d-P;5^e!P&*2pi8QX5cb33qX36y=t%ove@JUKfU|qPRLY4%NGgHj<jl1dZdI)T@o>91m`-Es=_E_%~0>*`PxsCTlF3q~LX`D35KpoZn0Wz9Rp67D`%6$aY8P6zz5U?N@u7-59C8{7|6@HbFp%{)GCs6acf&B)Xh^W2ps$l`4Bf!*g$yWvq>r5w@Y*jq5vr+SZd>@~;0}U+QjZ3Z7Plo{O#P3g>x!Q}!wbAF53<mBx5`z>oUd^M4cL>~{b7YdWcCnv`!1dFoZ`aFcO`4|i%p^z?LZu;8g(B2P>o2N&OS2JDOn2!1#^+tD65io8>M<bI9#w+*MMR^Og4CBi5gFJ}06CrOP61v?DOXPeBoFjh=;-YBT&?(&-3ZF59)n*KG}3gJS^Ub8Mcht%@vo$fGF9gIQ|8&=Ov^{}zs)wMWawdDbNa}(^(Crb^z_gtC!VSC;M*Ta^+R67Yqy0`M9r`H-AUeB6!x-qB@aB?!YYJGA8u=uCJZ#S2=pDP3VHP}&3H{6`=n?3a!4$&nHq!3TH^TX+J<|q757P?rXG`4LdgKbnPwx`OIwb<cdUmKT#T1;2H!FHm|h0#NF4TG|ht9BMygNZ6EGxb;d(NN=?`=YzB>&aa<=LZGxvhuOKzuZ~D%3PSTCyk%+V{eo`uy*S0n#HrRQWq3g-mh0yb$uGo!~S4;nG8G6es$%}ZlA6~P&Ty99?!Ou+X1i~`?}Jo%bGk0G+3<7d&JCk&wnBD_{;h}px3=LR**FFp|N#zuSUG{P=I+uxHA~2S=d`e47*SnGq3Lk$PKysSzpyvnY&HOxc=xh#2Gj!p=Z9o6Zsui6EDM1V#ql$0Z{H&RVH}#e8X?!{t)WKpmlF#eQ5wQL@a&Z{D7Sj{`58OUNciC275)cbYECg%_x$YVS0BTFgRSUT{<e!b!H7V$CXcAp8EHg5e^;QlE^Q_+W)NINA|pnH-Xb*|IM<r1+%?uP+}%sK<Qr4$axM|pU&%VkX&8jT3~e$Ma9vGpJ`aPwg$NO<z+>Qg2Pzq!VQps+*-Nr)m*>jzhU_F-PN1x;5wB0+U3mHC`XLjLU$oAV_$rDxFB6Li|Vgy#IU`t52)YmSy0=&Q;Ue#N0eH-%mJ-$(|U4ikMTYvV;fY4<dgXzjV$|c#JwdUgnGX%906*;b{a3slCs!O$LgdiNvf6xNflQnf7D8@wPGB1>a}Jx@*;s|kaePWj#H;Q-KlSM;&N(dVPJQA8}oFd!f?DkZ#~!EBrca96w(v=(@%gr0we`TL$+PN&Xce#3{n7__E(gBFA)3(97Fo3!Mcf<|2}m(qvJK-Q+WG>@BudsO=&FVcw=-gHyLg5_#17+K7D7m`&2mgi^0fb1QnOsxUy=BoA(X$_$qYiomM_;+uuj)Ih4a3ylCa@aj@HxQT52WagN;CK&U<7w#;9T>(4N-=Z&3j+|}q9$&2OU7ssHTSHhIV@@5!4_T1@wf_81GW~VbR)4|F=AworMgXM`>a;N#b)lN#C&gm`RuWd$rbR}v~TUE~{G%Uy8TIPR%uD8?Fbw5nH%*LPGKmEV<b7k<)JbFAp#q(nWZnyjJ=)XyHr*WyX`L5k%7W|u?IFSS1%Zs67Sa0ijb!hEf8^QIO>?cp)+S@@<WieU(IrG7Few0eSyMErJ*Fr^WQHTGWm*E+6qNedyH0rG+SsgkN<Jx7<LwH2FT-Vy#?bm+fcLt>`_Il1#he6~|%I@EXuxEfx7(bh9e{+oP{j2;d)!>s6VW*5t<2YmOxs9Dfz|V?v!%8lVe_!glw))!$Yf$tIFHz9eYAC_=FxY=diR++J>rnKza@1I?Lz>xvo`i1mFMSl(e~R2b<`mcR<^GIr@-h-DVULwe`x3ecI87;U3NV*{i-za@xv0p;a*?4AYf^dIpBm}(DkY6OkgI)xf=(h8gYFhNr?Lxmp`IKcOBL{ObLB(27*D;F+QQ4vAfA;<*kyi{$a<RTO({v}YQ?y-Rkz)d_htHbIk~9E_e6NSTk9w&`QyeHjFfZfyfY`wzfA>RUj<Ztl-Kh5T{i+tuPs{@YPVZj$I%cEob#<%hV|qUimyXcEzh`~zejqfp$oqcXi{la0q)^2%%Z-dLC}Nb?h1Rqr{bq>9czRk@SAz`n-BMdJPV|zepgkq_4rHGXYl8VpI?;P?0q=VWhiGFH|fE^3-Wm6jXkIv5a$EE>F_TbHHLuU;N9I}TJ6rdSmj)6_q6TxzJ=!RgEOl~;Ol+(Ip?|f_Kg2+`|CH+?wrpm7JEs_N9efXq4PiC$aqmLjfZt;6m{*y=hlR|aXTNxp;osEkgT`hNc%uzPm7N4v(p_UUedW<n>U`g5CyJ9k7-38)J~z=KTKEm@~$+g$$b_^Z85EtTM^e+5Bl$5q`DOx4BJ;GKXuMUxnMRAW2$ZdBJju6a!Of*p-`U)-+VImDOXjoQXh_{Bwwy{YH%Y{k$x3;R{>^G0i?KO-iR}e)$<S4Ya(Fy{X>a+{%QYcY~X>VvQdmb2RP(+@wn0eW$Cnh?vrk5*?7#B+rhAMMdBvL*0l`IXCHh4_!UR*2@7a=-t8=>C%y&boSjRkyoYGKMkW6ZIwNsr*F*U1r)UJ@U3q|Xt53Qy+wZD#mJeI?;|9LdH4hmKjV!Nx(N4?$C?@z|wG|4NO}O{$??)SMc77}ARZG42h<v82mtk(<RZ*W#vD20H=y_{>4`_<|+4PL};NzfO^l;rBua9mMIrwX*cWFh~RchLQj*89uFdOOgx%-6#434KYZ~8va%(^)kj3gBn>pPqupQYOc-1(Hg1|iFTjkd3HWrUJo4#=B?aaO?TQ{vp4qiv%1Iq)d;3bXH_TXmBb)Vcj~l65knt<!USoa9bPl$>*E<5Awfa3W8Q(?9POHyt~38h9mLWA8zK6ArGQs{Ok58?qBkZL%S)#JM`${R?6c1~NPSoQGS_mI}dVmB$EmgyNU_PKMy}=)ayz=@_k_ao-~7`!!}JO+9W>?UUP|2ie0c@j!k*s5D@z0oeS$q5lG8<0U#VXuo0z6bG+Wwv!)mca5##OWGj9e%dNc5POtuez0%g-)ky+p_%RQb@}wG^tP`Wy4w-vz4QTx;d4xP*xnZ#G8$PrURiL#v<Jx!o?0IaS%wwlu}9vK3~@Ip`__njF^G+7HR4vE6Gl#akms`>Hy&86n%RGp@TRDhFn2W=gNrFc7Uj!o+A6J3j&-K0>YeVp2EVu=6A3uQ<*L3zUu{>I_FOd>TykRyu=9N<hp<aKhuuA|k-)~kAfGRXAkRN+cI#Tfu<c9uHL14H@6k(WwT0r$c1s%750`g!GJCI9Tm9b_zUMUB=s%B3oXhbN;I+GdNB5io>1_}W;?#z_q0KBXXfqKFay-BMkEm;HK9`H4pT(3$HD-idh9snjQV6|Bx`;yPUij>f|BIjJ14x&1&R%Qpwb#Be!AX)dR;usLJ9@;G4CStOyX&NM2oA?(cmB+}bM=KRYVrS*tY;VsL^f7ge!A2x`9gEf`P?X6r3(l*N%rlZr_^pX+GaBaJI$6LdY-@IxK`mx7_suS=ZpVF5a_c==kL`#dDTksbw1F5zCXlQu=m;vcEIi4XHt89_tdC598&FY-PwGeksEPnZOF?M6F~+1SGgN2v-fbnx?e|2MBM|Y$HSVXX}9g^{7+&5EB1gJAedgg6v?6bcI^G$t=!(2yG3M49?RQW>s(!**yoH2*ppzRmX;W?(|WBPIFokJk$F(^!%}MqjRP%51>~;96`km?Oev~8TIKPdVLwfn==@-?`=U6VXb3}|e`CG5ePdf+1?OJDYvZNCh(cz`<~d5L#`IG$UM-kAV?nK6C`21$EX7lP2e?HJY1^@`2{Wrfa$RhBEnS3Z<+@~;eWE}1U2a+2Lc>l^>KoioIdv2OpU)6wbz3!^c9FH7CPEa7wb3KG4Qqp8I0^0y^>_{d#_7!s4I1p*73bEux8*&(2(Qn9>bb{Gjd?P?eJ{dVNO8b!1bK%R?~msw`*hPkucbOIJV%@k^LB+aITrJdMzClt>l$g$58Ky?$wEZVDBSN)KQ&<D$0l(dJj936-a*A(tJLt~ma-ztw_b0$2jcqe)UQtB@vqJ-sq0b$-V=DG622PGx9j@uxw3=azFOVmOW#}1jmn~5o6nNF?WCw0pv#QeS0vVJ3_NV@zbZOC#+rW@Y_0n^q+e>KqQ@i<HAIO<;YeW4)k1{zgbnUe%bQ#&$QZrvZyBC-kppSG=CIvjvCaL|={aMU2U9-e@BP7fcD*AAxK!!6-E+QxYW!I6^FN+=H-#2kmRCk;Ryq^ShcHEqXWwLF_Hx(_tne0=%nji-oj9(_kN!of9H~7**vd^Z4*l1sl-=3Gngxb#%Eha2wN_LA6C)V2p0R%UJkgfa=;v87<DuqiOTb0ZmTBYkZnb-nH5jiGL0zSOdREw|Iqet5d;-dcnBse`AQ&rrnmSC)5{w^Z-#7OMMDxtOI`X@<T!pXd=cQoldV}0ElkQZ28~k2qD<-zGrojoSo_qS8k<j7Yn*g{w+ZV(A;+I}HF=!VSyDRG0&gvr)lXMD$Ik=T=pOtj!O-nBZeX>}Uk}$d<`I}{{bP2FtoxZTr@0O5Fb2^lUxXrOW=z?CAW#cTnSLA!L=?Y#_?j4XlQzA|9Xhy9}maL7Q-nxcIvwi+7RfemH8-A2WyySbONs<-`lrV-~QE=ft^#wh>Q{4R}kdI!ec+t*j8+y*+<`je!cy^4#tX@bC(MYbC-w3q(lxP=hiYAz;fK?R6TML3tS)^)vqOtIC(jfWC3305-hiVe<r{4G+Yjx(aeknm^@aN$oAMI=lwoi9zIi5M%k<xfHpPwq8tqnTJ)eaGbX^q>O+^-2hwNU=G1-xeaN&cnL(Nb#m8%B<$MTEIWV~ybb&ZKl$8eu!xL7(yN1|Jg&W*SyUEd4r2_{@1e{pxz2GIubzlC*bHwn;USL8bZ{!uK<OeJT&2>a`lh@oX_viw8Q??XlbHi1QNvbUI&-(pF^_5`kgqz2Vfj>wM29S(=Zk67Med-XGqo0h%Ce-uct5RnTZp`FdMW>~X)S2FN7g{LV<-T)Y>FmKS)DHo_;Y-3pZh6`zGvzv7N|qWs}rZn{h78(jH@lH*2Rh9p~+%fI#@v8Y|B!f3{@rfpwPrA3UN^!C`SZcW+81{FIeG&HMS^=OlKHdPt_jbU@iVpB}kz0r2LI1hacL05=;YDp@3%KAHoT0z!8zP}<U4juuS`OR~`<NX}H2y-=8n&K9{@lJ4(8@QJC*8T8&(A?7QiQVZ$T#97%R9Oo<jNYt1R%L>%&`I-rYXW42i1F8<6-?XUtV2`#iSwMk+Df5q>y6GM(2#stZ2o!adR~FADhK#8iB6Vs50}uXkPC16HCU(08f(w1PoZVK0ylVYXis2EYWns;iJIE}S$R7-e~7o>{=O2gFu%4kU^-@dU~~Fdq>9=da#KLb{t~2-ggFxMx!!e{6&FjdFu^3m6`ojav^5NjWvEU$BoFJT;%+HJ?D2g^-%P^3c`ZQnOd$n<`PqIg?P6&<G%oH&o{tS~vT3A_SAhzH?%8#h5F2UrDH2p^Y=W%Y#tsx4lGmKck)8~I7#j1qwR!`U2r^qyQ##ywIvvS_x(({meAh=Cl@VF^OEAp#YuGVcuc^->5w?1A8~hBuKj2_jt1c*Xb}OhQ5bT1CEw&y?mB`uNSi58!XWEcT!%@4}U6m(~A4?w87<6o%$*3Dptpo22Ff(3%1JT0Y(OBj3gYUCOLzw9IiG#SM2fUhg)vS*u0S@O2>sZ%y{AlEb6$`>Xm^f8hJYP4Ll&W92Xv97Th}|ezE}#tL>c-EJ4-l008{%-fXcZ7U&v=Q!IN`5zQx3_-A(j1;vKksGnG?;$YB)Y(YmxZpXSIBW&}2*t)(|O|xhkaAG09(+;`qB|_qW+udq6~baGFndVaP<+$qfO{e;N3Ys~^m(=YXq$_o_9Yv%I&a?3#R3<`~8mDa0nBtxBw0;$^Ng;1^IP`Nf0>HLF*BuLqCD85y*z8a+c#f0xus5aNPen6FZ;;0o+tq|hU(?@&=Gho;oc+2oET`TSC~%&t8pVkOm34O(xE*1hicwWS}vE|=~lq6{@-wb>bFIJR}y-4e$z)a86oW9+ZmA74WY=#!A3ZhP11YbyDrPSu=A!)ycH)m7PxE2Ld-C+jG?eA2o-@06w}BC}KGrk{Ta{b@WImySil&PXiWCE#dsYK+t({gInpe5gG0<C$8Z;hhxHYG0n&?QBgA@Txfx`r>NAcHOdak3w{b=bk#P|Hh4EcHSK&ezmGk>O$5^v9PtQGOLUTljZvRP*;>g*NcsJ99XcW(?_v%VG`#f9(`ywzkjb;z8a0bNxwS7$gUKn5&Cb??|KcQwOP&ZlyY=P{!pKRnEa-x1pcAD`6A+7y^SZ|GflS}0q4l(3*U6sw+iTJ-R3wvOT{F_3f4lqzAud$%Wt+^dRlQ8Ejjs%uGm&*m^Y}Po)weyV>}FLWLW>p=t|vp+$nD5=(fD&$gZ_y>ei}0e<Wq<ox8ZVzf^hPZmg}e@)6U8bmLQH+{nn7Z4JD36Iu%Rmpxfl_<K+Lp0j_R>x+W5USAfo8GH@byYo4IkH8!C$nIN*&Zc7%x%2M}F3nhw;-v0ATL|4o_y}rF6gHowu6DGH>rZhK_Py*=fNoV{`Hq(F(%QAwW}(yJ-Q622<?S}`ER3P{OFy?0zR|-W30+nQi>E>5eqt)u^9&%*KC8iLRvDzk1G3wNRkZ*f!Rg+9Y7PFUGhMn=an#&`r?{6-r=#4PJdf9Ni5vE}J%o)f>pec-Bx8G344L?y!?-Q>nXU(_tssbkX;?rqMeL^!qC~ykL{-l7b{DUkC+8N-*a8G!yX>DP9mb9a-d?{h!)Jn;D{3rxDgm&jv#T8L?Q9(2Ol1ZXhF-rqQl27wos6<R0I9EKhP4ae(DVwOCaCcTlKcrv+X%aEqqa%j?}H?nx2-K0p>WYE9sAvqkNq<MX-<O4gE6hcaojNrA0yElNtq|kVeLM=7YeswIf@~(dALTUNofTth|+0Lls59>%S-Ee?Z0$SMJ#Fe&O4x}&$D=kXVSJXA>F~|HMm~T8W;+w0)2AebGmJtU^%i{$K&?Vj~C!|MRli@uF>(d`Bs^x@4_CKM7JI=>)ef~EcI<<h*S@MkMtb+_;}UUwqgqh$FJiz^!e^HUj@`Cxq`XeQ)5e<QLpU~8#Rv!r<z1|tRzoDoaL|PMZFT7Kd?<4<gPP3uk_=Q_L)4YMjS16znVC6(gnv?#(5p64ZaUjkb(AYey`<NA$>lQ@W(ppVlu-=!^t3Vw{CZ8(^V>Xec}<dmYJp8YAv>v#A}{*h@yz^R*l;E6K)T$**e{=haytf;QOK8!*O`wEl;b<Hr(C2mFaY)Pu#%Y)?Ide{!3fVpJ*X^B+!F85`nE`_4BiNcC9{o+AjqRoucDF&%%i#kG&7Gx~|lm(q}PB;pFZ<`ffoq4=0@(*VH7t?vzN7w)(%9I=*F>$)<(gb9apw2I#uUq!+2|RoKwYvs30L{lrn1(*?**a<*Il)ULDcy<C2~#o<R1ZM0Wo--D?(`l&Zvk7dU#y24)banNB`?s!)1dtBwUsspadB+s$jl8AN2d8ZTPk?MVPndTl(p_5;>q5ZyIi5kfC`Ol!bgIg*qv?iPMQmM??%Y>hurzk;9(!!ppFbgJ_51mtaF|CK5u~t=@uA}pJc*K5}a|xd5lID8ySWU2|nH#P1UBZL3GZ_;T_ue_e^`=oQK;4x9Wk-4zBGhn6?{CIxdcJNK5qP2p%Eq7KOn+rfE@GJ`;o&dHupxFG*2wcz4j*tOE2MLMToVPlv_L;ypmP7x_@^cY{JG77g5d!GdutAcz#Hc^*(B@q4y>CHP<^!K>7-i&6O%=aU&3cTJs%t9t!f_d+GzmWnK#2vZlhV>toKeUCx-?9%>e!5pb#u-fmNP#lv8(npEnXKW7$(T=#;C&EmDU@mj@52m%-KVZqc8?W$DVUwV)ewofT7Hy@G3$_K2<)5w3CqRbj#ycI&wZDck~?__lNBiioX+$Sofa+2&9OZzWQ{Dn<$FXK3wAUAl7g5>0~9d6Hj*&)0h}{F5%9U2aslm6k?&Ih1<!gj|Y`d0g2!ht1?;RE8pMnefN%2>rTfYK^4CK5pukyNKNl@+)xNH?4{MC|Pnikow(6-^g358QT;FngF4!m1?K}3=XHt#iM6fwHS2oh1+9eUxpoGuHbEofc9{aTxni|$gbOO=)v2E^FA`hy%RX+PSX7x{CT~{*a0^mbf&!)yC$}fnf^^#wAs;M`??vI{;WGH2*(NI)L)c}vAu`wDjy*oTq;-<8#U5sy69^WX&l??Xj$ro_GdU>smXD>e7iq6${nfIozwW5_TyrOEDgc*XdEK3J&6PoV28k(0rpfe#?=Pw#9z8Q0H#<AT7a*|m1-9nt;zH15yCk#WUoJd2mMY`-Z}-xXCbMR8ox9j@ZKmaC)BgvY_Lp*slYB@F*1Wn%4T<`cO-k+;tQn*-fhI_(>J?|bo+%4rOHqE(R_b{TH9C>iCk0T7Cz_~jkn0m?~SZ!q6^U2>9a0(`7?Xu@3Y?1eaI~$EGo@dDcr;{PTbAnc%qvlb_h8}y&v5sE`Cuh3>r@kNU0VkBX*pWHkKCA&1(;>!SZlRe5l6IdtLWETRvh<iGLy8z}=bT>u*;tkyW@!@MpJjKM<!+f2|_GtZSIpBe_+_ubMM{SIO`Gt<5|Ymzfi}GMegvb_(rk>DK6_3(Oh$l}4vkp~T)YSBvY)cUjPB9^2sZf3e`hyKL$X!)6>r(8TmJxi~qyt32>&WiyYNcuU&HT^8S()O$X-h1-ePDGf`&MY((+Xxn;<T`Tg{?@PSzOzDU$TZ4yN&qnHd+m7%2NW5o%HD7H)T2Qm~-Mv!(ySgi!@a9Cp3V`otXe;KVQe%|jwkVw8tp3V;r|;i)1vFf<8ep${E2<MrbcvwuyhP$!caG!O;5J-98?7ph=GwJUBA~xXv|RKM^eLM1JLwnEU~T9ikgZ;&Yc53TRv6L^vW}LfF7voBkqlxuAgtp0$-75ehUHGM1R^n&4!JYY)(hvNi`%4gs*AJxM`96aoZ3Jew!6Qqk9Bw~EjuRkHV75)`AMi9#&76G7S+*pCY%q92)!owVz<Ptbyk?JS_k{W4vk;BwM}_RBAli!Q9@Uv^x2zl42J^tCf5${a$X%XM>Ilg_1}o+`r8+$Pi)XCIh&5sm+qH@;r8=BjoeySYtd%5_bE&jCkPARcmGkS?xMQ=n*WtnP8jEGx_)u|FFjec6)y_G(J`MR*$H~(q-4k>^(3C!Zhc9!DRFh##!K?ZjQFBVhwq7p$ZEXa%Ju#|JXDD0p!@BM1!6Rw0vI}nn=L5+GfrvWhU#t8S*(hiz^(QasARk#6j8xul-<X()f_v2x#Xs1WmAluB|{*8B$zT1hp9QXXYylOKTV3T`)Qn%LA?OK4)Ns56%=h+_Xp$WWZ#(Qw?E9%$6iebsJ$yXVu5S5yf`-0LT-@h0yV?UJgMh;9qD!_dU>F<P+#9P=nM+S&nX4=E&f71cf?u8d#|=}j4J!^GNc@+DJ3M@)xa3y&64nEKT!%cAFE#<9IfYGW0rh5F6T#jpMI{5Lu36GJ8i7W!>x8)8O@4O<H5?c%e{GWKtDfF0#amp%cPC<KP8PN4~t;h+?QL2cV|og%;Jz+KI!RnmM!H;g`Yv})TAb3ysZoP<$I<+5DVyS_KDRh7S27fn|*M32urmcY{s1qf>pL(YM?YWi#nvysBRcsL%BCm5SiDxXs0G!tq%V+?hzV)T^tT}cmdYi{U^IOytTit%noVfZ>MPm0s$ntiAPJFh1_SV5513vkAzWGWq;~YtC^|KzA7h+1#ONNNL%l;(N>Ra8_ly3e2=nTU7hI_sI|nm$u-`ce~|R9ylx@Z|EGZ80NQb<y?kJuplxI#IVM_<wl@`9IG5yXj=^p#nknd7zf_xJ^k`i!5smj>`|)8o&fp2OjJB??Uw4OadV_w#MS_8j5xs+Z%N{L&_4%~B+b|pd(GDKwD~d%ep9K<hBs@QrA5MtnBTD#28!oYz&#+L%pS`!e-Xc_Z^mk9#@Gnj7(Wi^6+isb4JW#)m?tg|v1dQGPPuD@Z>>i>ipwtV%86k$P)d*qkdw;=IZe#e%{GT6Sy|0QgD%ca~(>uKhqgR1eXNl&P%(I7HRj2s4!gF{dulHxF_i^@nS4ua}T@pQ&jphAW1&HWVZ<Vj?UH=q!TL_sc<!M13pA@I!hrV_cLmc)Y+BqYq0$;>`-R^2rTr5Wgy$Nzjd#FzL@7B{DI)kF-h}#R<x&;+)wFv0ScIY11%6)YVY~K@?J-hB-UX@Z^VxFr>r%l}^YiyH?%V%q~N#@3=g9G~@Z5D#d-z_%Cg0Ps{#3QPko5=rUPSOVo>_S1SDw+xX87U1e$9qk7dPsG8A7Z7F*BerkD%!jAms?>25c{^>bIi@MVDgWR5u*IH-n6_??YeE=KLfh>XYgUU>pQz$&ey(aYG+7o#igEn*%GC#_c)f3TCsEbLfS3c`zU|~nQq@ayT5?AF!XaVHg{oy-3G`e$*rJ!PFjssP;Aw8P*7!WIgWa<0<kx`@!9ynP)Jg`>-g0Y0<&AMe6{`FW~nqU^sD0R`a3XL#XmX7XASNm&=emHr7;2t{bp7I2CX9Y=vwO+SI1auJ$$r;XL7QCU6{Zg%a3plE8D#0-F1s|+i)*OcOtxC4lF~li9F?>%>FT~3DfDUf(ioi*N<bit!BQZFT3g~$<p=g49?btwO_V{Y-}wz4dm=q+4eW;kY3~9CIFyY({xUoku~gAsTO77wK&u-JGXc8UI|Ke`HA*r<y*P+T`6ZN>=!qGZUpU<0tef`N?+ARdAG}adx?Tk`=Mp&?)5xepM3y(P_sZOwk$G($2G778v_Xuy7n~tG!a{$E-qQ@bz5uo5O~L4mE!s68!fCpH^+86@i!4N=ok5BsVR4AdE@uZ3ZeQDk7G-j5O3Q?UVHv9XObcE*zo!RQU;w%zo4!j1}b5vgf}BI#r<VaPo@Rz5RJDxMP0SrslDOqBS||?N&OrUfWlwRm83itaOcoF{gKs&Rs~%BeZV&U@a(1JOGzO3Ivva>Gsa$fyX-t@S$uT8vvx2`3PAYq&t}|cNxuRirG4lDF-tZyal*j)P&}(q!zUrpiq7>dAht@C^CRGRvrN>n>5|=P*ZYs^!3%v&^rok6^|=^A!MnpWXpOn;P{0&N$tg<W=8!x<$fD{qz|!g+C$IuhP-D@e4{>@z$OnBD<o@sQts#E(J=6!*YsAcq&JM_<{?J#+)hVyALv1k_;BkvbnbtzJp8i@h$wvIPowXWiG>jzS^+_u#(ityooT7qJUR%Dd>v17<t#ZBc`;%1F@7(d^373=fDEDdCa0V^B8jZ#KXWuTJRH>J*=BFrQK1*=H9?M5!AJ^2)v#$qw(>Sep%^JP+=@(WHizW8H=#+)-&vA0t(hCwdSoXHaog~a$tVx|!xeRWwX%@e6qq{$%!~7W(U?220R55x{N_+Vu^d}t=Uf211R~nM-j+_%|Pl-<TWp8n(w=y{{-ueDX2M<Atm#%kgS!tPOevU?}GleAq7O!<gGZ$*>r4>${jbEk~MF&du@%N9JC&A7Ki`gGZEF7$bzcv%Z7b{=lVGj8NhU$eyt@&l{(ojzKB2-X5yx}<V;V#<tF}yy)zVZ#`)16o!0~mdw*-7IN-ZBi+O5rA&9}Kf(I_6|yWmK_E{uuzpdv!$acJ{kHidGx06opF(Tru?EkXNZ{ipY#l6y#3IuGHnen@cNFO}V?(7c=w;PPhA#cBJcYp)&9K;fww>gwvvdopN_2rzf)|KdY<fpQ2#^7M|v?j7+M^s=Yn~_33#GYy5=OvG=y#JKliOD&8F5S$c*LKQ)M7%LCafR*k!Y^c)`>Oy-@%jy^RCtF=?V;7WZdAaJ()JWg=@cofDP$r+zP2hy8)@pH&9pfR|C5A3wIy7M!1Z3D&DM*xj}MN!vGC7Id4c|-io{9NZaVUebOtp<X%72raz@7n%rOH)3R&CJ8bC{)e(29*kE=6IvsW7Y#NH~FzVj3}*9tRLg~lq{V}_u3^no?nM7u!E&u9J|a^r$B}EWA_hDS+jZ0^P>?oW{Ko>uC(Z$QeNoK`y`#?E`iVb%w|kUezY0NGGlb+$9=umQPpj)XCaGTQCy{iR)swlMg%O@R!rTKZsoBtsxtE$S)o8lKK~huP_-AnQA{*jz_?@W&EeBC5dT`8kqR<A9Evr+b>NojE6c3$W2bcd!r1OHO>Sv#9hADz{?n+`9v-!P8O;uLT(oCUfVN5YpxL+fd<;ET%Aeqm-#4&ywAEaUId%wG(qo?2kc?jj&82mF!1@C2WiI^|c9reSe+67;c!ci-WD+|Kg&qu~^1U2LhX$M0g8l#irfx4#dvK89OTQIuNy`uZ+%Y1Eo&hzr2(|k$*=Hsvr?I;qC-n@hbefU<0;*;45{NG%OPQfkd>BO;N3Ms#r+n;ADx)T-9OfBEAMeWtIMmxm;#4lX!1=R;nSnG)Bue!Le+rlPPU<cvr4^uMV*SG%gX6N?)%l!U8nLG~+B70{%GrZ?&Xs)FNE)?H@LJZJwwA=)#X`^<(tIECbKUQlKTJhcf>FP+g*yGtnMK8RNS+G>)qqBX*s^LWd+2>3l7yCwqEj$tZrksmmAI%?E*#ac4k)r+OSw=$II^(ap-e^shS999-OHCp&^9oD*`E(9J)Fs0(le)7YVl!E5e8K1mp+=KAstTc*lrRphIrlh-5Yl4#o-(mvgc4Vyf=f2q_w6Jtv@QfI?W^HVli8XI-sw75gd8C+@#G&A5I?Gb$r%4X@yvl!w$C7hT7VtH}|HTYs`zQQ~R@-e+J*!d1e0k^-lS1SibwCqXxhhaWAC9C;OwiGNDZ`Td8gx(@%fn4dSr0<vJe_hA_Vg+^?jDGE^{_OOLm1W8*6}`rs8dWJb|^m9O3-QYG{L+PZ;jgXpfU(49}iUA$W9rByB`>hZ;_{w5s^3ocg@WJ$D+CxESe3J{S?o2MeK4WXk~aWXAo{_(&JOBuarZf1FXEWD8Da-EOQV&fZK=NrajgUNiMLUB|dyLIS1LSsvzaCdAUtay(8sRyjKxDgFg9gXe6&`~m^n4q|zYj+!WJAQO0t9ufq^WCAam(=H~gv0GFy`qFs@IEfT{I)F4BENs~l)A1cYfAoS*ezE4<u_pVplicib07Fm?uu6DL9R~=cb~h`XbV(JLF<3B*IUnjsIZK&*!GK+xjPt%tNmK63+_WH?vSlBP-8884+;JgK>53~W9ZqeecPOvmXv#Dy0}i+v~gNiR3MT#SG{$+_e-^sy-!?)yu2{#_-nsQ$nY-Kl+cndotBai{rkZo>&sGb?aWTcI(K}%QsXnO6zp+v)t`-H`~-sRGrAb~X=H6ym#1Pd2az6=4XPu`E3erp91B_FP_P%fosu^H>M%j)PBw3?B~|CT81yj3XP*i&M_hz;;8qk5<L@bPUN^`Fx<skPdLuIDx&?=#OI5?MC%L!x$`j^$782g^gKv5BnNY+G=M;qN@?eOFP0RY*S1twmMt{PZjd_9Rki#(d7*-;S*XB%>|K73*SDlQcVRf-malYA`oFO`{I{ixF3UQ_R5~z8nn8e?~_l!*-a$21ZO=NnEOO2v)#PMzERxiv?<NmW!gBPYu)_OFWbk#lLVD(0>y1xeNqV#qTx9Ne#YeYY{&cJFq?1}&$FpbMBsrC9{JS?T9ezb%EIg%$(BrH?~2yRo`DSRWZX?CF~=3Y0iK6yIGbD_Qwz5qnnU|ntVhIAKafA-*?D<}=Mjk-Rq!FtV@f^rd<R(Q2=vLmlmtQTrK(}Y4_02I$=MtNWDi^Txk6Q|d74C0{wnC%av?Y;UXKw{FFE~xI=jeT$4yN!=pnoIZA->i-Y`H1v6LbRS_C)cR5rZ$9|UV58G<MKY8bzj|dChH6sZ^Wlfj5=`VLCLw?`hHQnKRZX<CiQ-<EBK|;<tfM-Q^SKafHTnfVsJpVD#;rml}}5-!?xW*8C(LF?AzXj2YPZj@XsxJYLP^vZQ-ps#XmnUrXz6Tty61^2;RPQFIpjg8ygz{WBadUF@S6Fb2y%@1zAV=>V=<i-7R8bq!YW_SMT#~e1tu}2KF?$Jhv-t8Fb+na9dD^(t?`7L;3NqI{y$7{Jw+)<Rj;q3cnlFroP5<_<UsIpUpOY;pSuau|3ZL>9M)3l2X@Uo3}9??&~9bKy(nKKm9!0pSE_g!T0_F+VH*r+M8O=w{<p+;ZqW_;7$sA56V_DitY8mqk|$t?h~-#?O1pJauonoye68SqTgF=jG@mLo;YA>O|1eCxDTT6eINCB6)F<RfQD|FW)Vr)-tx6>dRPAUiM(t@*8k!OfW<~VMW~IR-E{h*akj|{gJvJ|R_Gd@GxyabUn}iANfS<9(0Vt=;KH=?-5TAvSDuw=_(?+CV30aY(N7-}%z6Ln)-w|Ab)k-LdsLq2<pqhtyT+4>;~jkmz^1caub%2ExAqea2lpXE)kx!Z*FO7NHa=xHG8jpd>|P#R2hI6-8a?M8=rWV3IHRKYQ<ww~;7`C(z5enf!*(}#nXihra%;k}JvC>1M`PG9hYs)VBNp^!E5eN#AvL4@D110WOqH7vxtUxXPx&y1n9K@5b0DSOyRaPH#a6aEExP=0%^e1J#~5(==1)2`N%iHv)1?bjrM$mXPu<U>b!ewViM@Cb2N(N1F)wGw{>h}Mm*w;fz1aue%^q+e{PLEI4}4{{F7f8lU4=wRLyA<*lP{#!n1!b?vHc+d_X$!pQQkzHp4B^+cPQZmG3OwG8ZHanxC0+0*;{Y+By-8ux4J98z9&oYbk34G9^JPZcvO4ZP26gQp76HV^YxVUXX_CH(Y~<i6@mr+Col%m23{XM$248|(mH9e28L@1slF|wBG?^u@e1|BRR#@n^yw6gkzDe0;Jp+JWN&{RKZNk4f6s)2WgN1{9n*vsO}q+Lx$^Tu@`TN-+cjlg)Vj%GqLQ9>qVZ;EtpjBt!hzlk^10qhcb`E9aGW#GggvIT@`)xlWe!QV)u?%r``5<l+;x#!dv)oDP_2A~?6_=fyRq82{N>FjZ*|`Xl-9L~nPMr=cjwL+y=keKZT)FFxH5$~>60FkbojhdZ0yo2L12whjk<^JVa1(GyZz@$7MtCq>$2OUUT0y#tZ4f4*efOr>5DLkKaJ2<4u{3OZl07~+3cUtV3`y!W9~L5L6e@3s<55On4fx&=`SL!j_(q@j`gDCNv9SMUB?CU`(8RgwCk^nW^0LO3cZJvnKs#EYzgly28If>XL@-b0K?h$%E{S5xptxt=UwjJGbZw59%ezmcWSO~eXDAf{t`Z1I541*Q|>oQFbXihY73s}Qd24J>0mESSVeN4xceTS2Q|=Dl!ratLX{o^5ISbjF*w~AjoDya)T+YH-C@qJUSH#K`;*<XhmTV>sZEPrAVzKl9$CL%%m!(eHZSkIhut_T4kmqIyU)jd;`+S+xpC8AP*bfbeecxo!E>istKK7QV&RK9auT=w=^z7hfqSI%nt|P(Zt8w<_h37z#_cN>gotSAwx3k%{OJ|51^9d*ToX4cDYYJPJ`d>G89ii&Vs&-z1e=44!K0<wfq5ss;uOnG_l-?x7x#bJBF(`$!j+6I*Z1Ya3aj+(lRNAT6)^|LpGuXY9ObiM*M0Vq=YvTtsg)F<G|4l?7d&gry{TR>Z+Fz12)-B}7pJXC`8E=#Q+L)nY%8=_`kza)U4NJE@4HrRq_5u6oUfy15~03OoiCc$L)G#oLGT7P|K-x0mr74-d=tGb+$oYLM+y!pfZGR~#SE}1s-HfEQPoz3i<mYVI8ih$4$Ks@_{;i3+ehDgmNrXp>bkH^e4q1LYT^``D6O-Y%NMVLBn7(hW#)@@(_Ogp2YzlEZ)3&37&#_XA4i2wjFad2ax3$>zTeKWp~lUN=5@8sto=#)Ytb1^D85iR*Sf>U5x?6?I?Z;&b7{pyckKktR!HG;A_ipkcFN`;;J!!@KC(Q7<hRsc5@~0%2iA@!Q00$RE{FSBEkQTpYjZj8pjz|)Dxq{Fg--_ep3S`4m?D!7g;kc#+rYJeyJ;Lhm;Tiraa~%HmHkuRJ6(J+dS+{v6?U)Wd4XS}%47SvIy<-Y5Qc=cuT0cpT(^B?a6f3|;WG9A%2k&K91qg2pf`qH?~tN))!e<NA?P^EWxXhmRln=nM>wow|3;lVm)^>>dhj+H-8(4i23~ExWiTqL-o5|{*H@?EZk<0nW@XAmjn(F*QgrvPnEVJ+Q*_^-PG?gISrU}p#mF-IYN3$fQtuk7XYl>NBvq<aEsw4^E~p2K@F5g7NLXLqN|#TM=tt)fc{-18kDUeJph>hv-HaXS*2O(TF8!6r{S!cf<OL?nj;(v@7yaISv=v_0W>a)Q$o*op^Wl`XYRh@AatJBtGtReb`-2VDkrvSQmu&4FeLUOGHm&R>)tYwQmBz>Pw|!5W*?lNqP-}u~9bH;ivG`W^W|ZeQzTd6C-tbK#yQb*w(F=Da3Qv-8>*e`W_n8!GU;~=(y@ZOIadnE#4hBgGmG5Hv4X4klR!1KQS&tw1&1S7}^dd+%P%fJc=H@x=LPy1rbN^h64na5TERC@|uH)ZT0@UQ?)_H-ZwA$~GL1p2AT<X@aTklf`nEeP#S_xMiEPlI>$!GlMbm;^@Z{&<_E1!NskHWa^8AZ`O9$!)V?AFjCRwj|@{VEf?DyY`qrMqubI?0(aj{|!Ka@FJZPER_gD&;xlxk)_G5=6bCHsX=&_0NQkRups;MICS<mA;IQkGBka8%5R8zCD!_7#&sppewcnp~Mc+U8TxJW;9!67`)NnNb}TkrMdCA-l&+Kw%|RMevfm0jt%g~m^rz0(&iKIFBzEL$<e5+jt}#EQry5w)R*tr+TWCeISno=&54a^!9nXdILIEH7_7RBXN1K_R2TKa4@ur_M*Cvl#JOyw`rhQjhi=8{0G;$~h%MTy^MZiLNjtBR;r1TJ5-<%m&-1T)YUn3VK71XYsjzQFz{$Q1t-L<crmd*2l%{&UblY67!-3tf7zXY5i$1Yfw)|F;xSbGE11e;(r|ec(hg;<4(Z;W5xLwXHDbXp7{UwgWT|;EBX1_%D&-L**K*w`ga;h*vBV!<H4U0yjxeO#Bd$r5K5P7U&qW0a}e?LFukuib0EHJYW`u+vw>pFU@_q39Ic<@Na;tUbA(AO806v9D-^)df+l7ZIH2Bag9%-vr4b3a9{hB1PbhB-kUsKe52-4@P)noNm#VTEu@buFk9Kg7Lln#fi%^)q7Jt#xXftlxYkR8c!T(C!#pUvdotMR_l!mAC$riqgy@k=XJ6>^7&Bb{r`8`pjKHzm^$sd!1ZBlJf|$CL#77U2Q~{Mka}<9Cpo>DEZ)FTk5|W_M?RE+^ZE1V>Y|-0`D=phK;9!cDbEu@9k88=h@ua;FCG$sds#$b0ucj(`KzdvHuGlp{LJz5yrSpubv!}hz>ZfLZ_;R4XSDuV{Q$N70joPA2pV5WV;|H5YaidQ{dKj9{YmSW>_tAJLkn&gA1X%KT1Wrvj;a5-xKQnnJ4`Q>fp$+-&`zzF7s7hPeKlfN^tQO)SI)$`^s0e(Wrfo7ud20Wt4EEiRERvW}OgC&0`}kar>Ld7lW#6Z1coB&QCXftOE3ba4z_vb?Qy@+ipQr(@}kA4Q9{3>5^yz0~%q?I%*Wvx&iMn|8mma8akDG>l@v;b<{%JdLIhjmZ<}$pDKP@ts({4_EzDNQ_h<GvCZu^H@5CP8%s`6+D?_5c3M39mt2+<iWoYMAUq2{&B1AM+%LRnu>dVVO1~eeW<v^DG1B=F>VZ>v|2#&?wb3qJt8h(iX6;vKch<NkB;oZ!G&IqOR4R(N8F;ziU_4HCbKcHqHL9!*?S-yCSne-N9wIW53JR|*;oRG#QKb{ryS)Y?e|BfE|EL<*ZELN*1;c*5x6FW&e0{98x0qGsJBHdHIg_xuL<z%|7E!@xYMxQQqwQ0Wx<#bhO)s0|@t6BA;#)B8b@{j49DyjEDG^@TWuD-5>zod=KCeWp`-rKI=eo-u=VZ}*?2w}K_MZhR6R#74`q$tkdSkc<rmQ}?qB$r#U1mZx4b41VO_}+tJXh^@P^uM2_1R={l3T-J_)`Nof{tG&h0fVUjXGJX0_V>F<gjtTg&U{%;VTgGwyb#7%O9VT^0>U-!|sxn3d`Vq8xq}x%#fS?Fx}T**@tLQjk!3|AGgD*YUIB<sF}ZE3|Ov4jI59V#lN0?$keqkHpeM%%zEdXfVya7ZFy&CY_7-Mey`K2G1knx9r-k$0N0mu7UiGBo!I(=Rd;nvi}pAmZFEwzxXID4&Ncn$R<*OTh%VFmwWXHg<>UVvWuG=`wTp$-i}SHr@68%UabVdC;Pk!~V41&(REHw)7JB~v`7t9jW%HS8u<fZO{w_Z1rJ8W;+4I|!bf=`~vzF0O(bF};UxSIaLke2|Pg#C{orf)8*fe)=(_zp;K7t~&Ufrfw8on*pr%g~o{Jn>fwBfj8m#XE_bRclG!(*|@Q2$;XorMT)P84mnxa^+|J=zbX#p6m~(WwnBHhsnTgEi*$jOONNSy|Grnm)ro*O!b&c*d#iJdo9QTvO9$xPO?gb_Go0bV0@f5EobY?2<8SOy-R0=U8zj@TmUz=;@gr#f>dDY%yGQl^Fb<4EleY-<btsw(MfxsI#BdiQY~uZ3-<&g!1_+t;l`<aE*RJ=&ekv%|%}xv?{A>PmeB8FK+GiSynw2eeTqLA_NX>M%XF5DEMpj8D{gtX6-V_N0B>ek_(7At974@-A@!&hbIG_d9Y)iY*>#Dkw2zN+xTl)Iv(D&^SJ+9Z_av!+YP?Q=VA<*_9@CT`{Avc?Y9$~&XXm%v^;n?>nI$2NZBx~J8<f*FGGd2Ity%v&aRCbGQLTrS#}317n3GwUvX|Lk@`!5j5gI<wZ4$byVfMy?W&y$SMIvgSCjAe(|4)0#b@JDbhCL?nnS>vn*3~@)6p~k^szd4ExeDT?VvH4034FDx7x=&ea|d)FZRs&4%x4^6jBdY7@<^+uqAfWaDVaYC>cuncB)mB#zjyKR5gmX{k!D4T~M;0)%!_O8h<wl+V`7YGwjBBpvdR>m)FBTO=@9s?M^la8P!|0OGj8W8c0!@4cG)2_#I7(wEk<RhXSt^+ux&YQc3fwGkmv~TMt*CO?}h@F<n>p6dOOx=a`R9EEp-@MUh={q<yKV&%E(B`K{ib1jejZVSJ`ZO<D2N3FL*4^v!qT@Ul^SHs>~en*loj@*j!v!XmmDwXIto&^M;PT#f6i#w|yB!EsJs%cK4`Q2Y*nrWe=A=Cl|c*jcb_jprp8A(>7N_D)MpLC4buRK?j!{}=+B?Qcz54;?lZ#j`f%pS9BX=%wg(T0F3a<>%5FRJ@2<yh*QMoM&*ie!_MlI>5W8m&469dO-Ix{nxIxI;37f`8aFoSf300YHQ7RC2{%jCJmk)=nZNSx7$a#_}zcyi^>PfntW`@gVKv;wgDar{z8nhcv<N!ogQKj%1h(Ac@-xuVFFZWlUIAU>tAhN_D(3iF<;9Ae|(;DKX*v}?CXUQbd0Qk5O&=gPxr(UYW81t|H2oubWeWzQ{rvuDYz!P=XN#pnETML^<HLsxZl6^ccWN%$65~|#@T`W&8}JtvZ$`ay{Vy&a6r)K01>@x92cdbGH~Qp>DqTK|AnuD+80^XhI1unmqmJhUnxsu<`y3geS`1<P+JSmugSLan1;*R&Q7+vwCyx|rMW+D0CSb<w6?wUIIBgc-SfT~T@&Kh?|2Q@$28d7m8!@$b`$Y2tDLosd&nnlzqk^LX@6BcJzr9cCpz9NtWI50(Nn(mvuPd9Y;sU@_WV;VO`0}Zt8d6fcko$A?DU;{Ktv#5J-ud&6gV1`4ohfUw%0|h+3KGq&>9{d_rtL;x4B=KY!Uj|&4vdzGMxT%r@!`x4A+}0!U>LTL~f7~2@Lu#R)9B_Dc3c*lc79$E{;HKF&R`(B(5ytmb)c{#-SPStfMrMN}oQ@Tc6{NR88W%{$oV~|0o?q2*R;m>mdZ!?kQEHe*L|Ck8a_<#C^V-H$YBYm431*44>P-MB8Gnz?v>{6(Cg(ll4_17t@4m7x{wMxMl6*b=icW=q5k{U*zl-%Pb}B)FUSEo8Xov`&iv}b4W6jdo^HNHn86*sKM##8iSTXAXcRBy7+VJ4GO8SbR5~pqK%2EwVUaH^$R)*wNV?mG=Op_!UMeZ*?M=fmo@1ZQxQ0oW~SmI%;Z^F37#8+{<Tu;J?0!dnBM;TSNNk<;^A6&?Jli7gFTyRn&#>oZ#?ENF}M}h(7CM36rGIrb$C!~>Ekjjyc+#YBCE}v$Q8DQwH3uYN`<%bBPlH(C<s^oVzbRaDiK)w*(>O-DrfO?0VaeU)Q00M0k^MA>s8V5a29grc}XD&%^aJcg+b3Rc&xQ3(w^<SK*GlC%qY})d;<ig9M;PCKJyr@4seA?v;1#rdBtR|`AV*$wKU(43<<2$X`+dNlvTp}nQjmI9L8Mz=4E%$WmCCIXSYOm2d`g8_HyM8);k1>?#_egYBAjETZCwayPl1z>0AzbmGr>P;dA&UK>D^D&~zc@Ztuz4VUnof;X!#KSGOzX6FS`X<6?%umLlNi?b(o~&ipmv2-kUumD>dlJ4#S!)~lqE)Q`nV7?w0cifDM)i<QxWDwyy+8^5B=u7w4@lGNtK#_WLF2ac3#;jw!aGb9UE_?XGQl(iNSdGu%Dprw(!*5b3GQ~~p)UHK7wx!gNZ@<rbu42rO$Q&(@XMl1Xlo4{p%Np|C3J5IrCUAkaI^EyipI}jBOnz$qzruASH|90;*R#C`H(@)9Tdudc^VEFmd(pxLU;dZ-p_fvco4|nMn;F~S%?dt9ulDxKC@78V&@9`J}9b-~ui_7<`J?zvU%*1{99&FeHy0DbZ!?7-`P7Ry0pHog>&(WukNYkwDBru$KO~*UEr_ANd{Ow}1jIF_tYH~%R1WxM5kFniUa`?0~UKHP+0A=F1`H+k2A`G_D$nK`Gbh~*6C<YX&v<9Z@9HC#uaUM?>#!^*&Fcwai*OmY{3YT5RRDb4th*RCMakJ!u&F5-$;}q@CgE_llfGx$|X}weR{!;sgxggtyO?07ZDV+BJhB($757nMMW}_R&aI<MiPfRD_3-%|CU4OaHO*nLXMP<cb6`x#^edMl4mtm}alAxpA1~I>UO;(H~TF^51a*t1|*;_o&@mg#3Psjg=y0&g(*(my34AoGLJ%$Jw=_H2^R4C*kIfN2LMF;%$Z+{otd$-5F*|eY5Gv``!%~=|{4{vLn+94^8^Ygh2^wyc)2CGfJ)}~_dk=$q1^lg8Q6%dE8;OuHF>g#6dCnkQN%kpZ7PLc@mUb}EMUBT@#UwWR<XCC%M?F3mD)>|55mxiqLNJ^RbBlZ&2K3=YZ*aan`2A@hzhmm*h0c~of+uvZhwr3%@KbtY!Pgu3P#+R}nl_5w(<%K@tyJLf0ziOMsIczV+UG(C2Mw?F2hq3E^7xbh{V_clhfw7v3?zNR_p)OUOZ8T%m%)K37WzXASd^kwSv$}@k^|)9pr<CxT{7szM>jkvJ*U5+>qq>rcig{n-U3<P@2`UU>y485mOV&y~qd#P$Nxt6o($N&TQpG9u>8`@b<HKe6BX3s5uFM-n-aTQB-R%ryM#ddq%@=z56K3h_=+t(WB{eSJSC?X7<WbO9yc3e8>xsWOm^vO959UCA=b94hJRQF>x&{!?+)R&=cWQ=vZ|D!7Zepe+8xE6cH^N2g7HGp1LP{8v@a>lj_m`Vpv!>Yl48JiA#wzrR-vZ|6446G0z0F7|*v;!hNO|+Sk9&e{^Fieo6uFf5Q=?KG5z*1Sq*D2mKNCcRY*}%LUFj)2(I?%_m>)eG%mJX~Zh}D>c0GvKV+f}+2Vb};(^=>nFGzo$zPxWZSC=KzzetDSv_$c3kWBcL!D^NC^8n`5Yr4j=F7cWk<fGdy%+<Sm3j5p}8%5Pn5aGtezBKbj<p{}VpY5Df&v`Zm4Aa`RsSX~$m!-6{q8^w0oi@+UKZ3S0=*Oq6n^2kUUK*QbzwnoHWoN(vtZ=_5Y7Cp#(8``wZ~E}F>UC3{Nc1D9e0aY`bAFoxC0Q_sF4A0WcfzNx%PCPj3f*$2G;P)n6VUqmR$9Wa*L`<5x(|Y-73>uvgG~Mo)yt>xo9@n8GPw#SRRGL>BL}m=6lD|ZAi~$Yo<n@?1EJV`T&1!4;db|*qoKcajw2f>8uzLz1wivLZ~s}5|3p8Z;!3Cc;X5_1;Bwo}uKVU%mt3Rr%Qt<ne2~hZ*%M~c%KfGFgZZjlyAMa4sx<ALV8f$%a_l&(rzM10%zmAcd#$UPj5XNLU(ocFh^1L6L+k2>&`x!~wm5{T%Z@5CWYwCpbc?8iQ|TJc;f;fF^ag)D$7LJ6-lCQ4tUetHbdviK)If6TmCLlqe9n`c`E%xrci_Bk+6#?TT*yYE#Iu^fF6xbu{bsT$vVQ4W$zHjS=cI%nNUu9?whlk{T$WIZagrX@n2Gh_VFV$y&v#5C$5e-sf=g<^9HDzdg4hkT+t-!0Lvn9uK4YGHv*Y)no7jLr|0rH8OO1|ncQkM0Z?Fm2E{io}S!;0f@=e$~PK?gmj+S0JjrSntLj$wJQTVAfp>pK-_jT&Jr+yo1Jj2fNFFLXJzOC8DLmEH<V=JKF&C1B3jHS}YYA!XJI^h39go&xCrGg}vh#K#v#bOi@z_b7gT)qFVz<G#dRN$`t|MI1K9dB`gE0E{?zBUz*R{!)MbX9CT4=@v%Ci?Og>|z8D<Wf~vSpC#c_AOR@r1#2V%P$+`G(|6Eu0Lvq%Q1VauiihDbh?bUGx2){R-~imVkoqyOMRQqW~$LS)m4C52=7kTy~b!Svl@B0ye9`dZ;VAdF03G$yB9FsE})q?y>CvBJ({{?^=YMzA$4qnPr1pL4W4;k`#i(rz+cIcb8ZTggKlsXE4?AD#jj4|a7eDvyaM)UMFQla4&OM2W+<+DDNMBXh%O783f=nSC))S9a1Jf6bwWrxlQ!_wI@6!9F5dJ?1M2oykML`K3T-FbQy(kWlh%w<g!=KkM?<ayZ9lKi<XJ~9m!Iw4=M=v_TCv#bsxKzntecjBiEU216@1%(S|P8fUHsi2Kp}A&^eaxYxdo<!eBp0kr3(H9Gv8|^@<HK+XENZ{lAQV@Axq&*&Ql69_UgqwJ(oL^9_^3U-Km17Mcm74Mh2&FR0bAK(E7wG_ir!6+#R9OT_0uCV~Yjj9!+%`MeuvCwj1^P5Z*_!7Fls$e(u~R>L1`KRqyP64|XY9c7qva%!;3*vK5#2zWd9}6KoghOKWW&TtMl39o?plWz`Q8>qY$n%WXSpgsbbU>b;fUACNcydk8mNtr2yLtvjxEcLm$EIt-{b=bjW~oBW=?N9;~Qk^FG7%T?@|h<&i%g0o;(Z2RtyI`RE)zM^a&ox;gGxzdp9*o~HMI5SID;P0bOt@?2Ecr>v|*{a%u;Jh)?^?I|A2>Tq=`BLboR`bSdb40Q7dN}Q*n$~+id3lCzhr`h&0<<Ah{l0=jiL<P=nu-1Ex#A1v$#xfb3!Bt?2(X=DA%hRFu64gOhYir4RQtb}1`?UR@<W;*ha;A&ey5LoE*IkGHX7u!)9xF8#DO=r)!3bYT+bI9V3pq@=jcyo;!DuyZhJF-lVj&>cGJdU)H0B3VtU~dIE}SM>m=xdw>8twAutl^s~}y|6$)PL*>$3w2g!PQS2tIxs{UQeq`l%D$reY8wcpYl|9N-@mneADb=^V^Ui_nqL-8Yz<0${dFiFSUO=tdR1kbNtGfH<jV!c-1F=pG6?D73_JGu$~-0$n#V`qdagGD?7))ptwfV^Mc&|8gv#>=Cb+qQWn(lykcR^ao>v`!wK)+$lyc>Kv?X{m!9ar5r6V%^YxwgNFnpXbxpJG0-BeLSfIu&*?`?w~A){)7AwQ+zx>8MS!-o!stl>$c#YYFU#^ad~r>OmjMAb&{5Nytp6RBasXr*G~bsVF!Cm69ni<7K8XUHY10fYiGzSwbtzleA`n7*u9Ty2LZ=q^Ik_#d!vmyt!b4GVwTj&w9PgrE;x2`?KD5gNasnD^Xpus(#6(6(Zg1n6RuffjhDE+x!GI)!8B1CDc^ZWvs-P`H5<|=rHyd<g$?f=4Z1&Z@2m@rHMi<aN=V$PoY>0JuC2E1H(NG#jTS}^_V-_e3S8!u^>R7eChPj-{sU}?;L9%haxz_Ex6R@*X>i#?y+NgC%|(S7Fe$#_O+|R&OVX<BR$seVA|psk@9^B)B)BFtOU?U=`F;81>hy@&4GebyzJ2k|K*Spo+FZ(+aNKNXGPo8>;V-aZqBh1J2nru!WB;jE9-?E9w}ZdFi<cb`k;ZR?7&X?;d6Mm^bC2^{NJo||-j={ynLoluR&M>GEY0q|{anGr&o`0}x$m%^G&n)J_Zmv&&4x+~TkH-Ad_x{erw3n-D}{3k#WDaJ-3H7!hk9_Q+LhUMYi&D&71JiK*DP7qA#S{L2F*T0Bs0=z*B;G$*-fDIPgyR!ZBTD>q@qjsdAGoqtJKK=`%C7-X#-kNXMji}ZhPL`-^sDoHItMpIiCbZB=~#`p@#9A7ykNE{>5;bSKWn{8qRTU?@l8*85pzZ;lO(WeHxpl?0%&4^<(&)zn>|m^(YqqCZ92i;`ghJj822qx;+bsL7QLM`_kGu;^;w}50g*1d?)86pd;Pg?)fIqexz_jqR+}_7gf4MMu!O5ZMjW+myCME1>u>X2KU+42wfJ%Bh&u0<=X<_yRPjHu0K|~$y$6}xNZ)?*93$dkMag~yxLOPTBvw08*b6Gb&sB|whE~_gn<48-dMGFHv*9uUM^Ka%s4F0e~BWMPOL_$EfxK<uY1L!^j?i4*dC+drCkxjO;T7m@lGm#-+Fr(4Em?IGsm+D{?P}h2Wv=qR%>U%dQzf~?wQL%i8B<pVDaRl`e^Oy6`ZjyRq@ujW<SNLylt8GPRtqZ_gJ%`u>;U*^--4P;b}*C$3I!>PC|w_W0Vf9LZvDDFlnv=Oz!W0iygZU+sfxhDDh$A$o7xiaDfH_BkW2iEg}2Reg#!eUzhb9l<JxZ9Jt+xUHL>eIySB|T~m{5KblR7D(W4-O=qvBv|MjFRMG15Y3?v@@qawxVod2%;n{zLgT6!Is{-x8$VjJO$ha7hOBz~nm9>MGs%ZLqKQvhSi1r85UdL|}RaCbbMOW-}a(zY(@_zc$6S9)R99K2s8ur4+zCml0Sez$o@KC92C)1YR@7E<m^L#vdb2V(?UctNf*;ln+7C-X2*&qh{zqG6UMmzHLCL*QaB!N8py}c&V?DAkPuk}LfZKMmDB0962oxFn#n?n7GR$BLVP*gbk2tAGp-n&okZ6#h|Z_2&j#7XV(5YE>I(Wwo}t3++zB@AB2J&|NTn059?1c^m|azzDdvCHYKlT|k;o3Sh<Etzx8QJBNVUmi2=I@{m$0bzutx#K5$t>w+M{gu97@oQyIjl)BB8G?YBEq*H|&Xub(zD~cCF#~N5JdCPei7{7g_JR7&=?I${J-f9*!QQ%9vhc{nY9Ewm;5X+9(iW@_N^f^ltU2N`W0dJ}E)Su~*4aQ%FT#;v@=IUbPOsllkSDgDQMtYvHk{rh84bUo^3(e72Q8KA8oj;nyNKQ3&GF~xbsj^3m3FJ04fS_SdIFZ6mq6Z>aZFRu&i%fU*yJG1fozW8jJesl%7$@JfIn=Cn_Aw}8PseD5kI!=O6!7LX}*Tb%O0p_bo}XrOSMBiO703CSFf!Vhr^W+TAl(ZqJ`|tp*3|xW(#cp*Y$(1-N_wTA9q|K+&xEcm$t!}ZBb`6-*QW$awiKsK1mP5Obn?GZ68~+vfsd`4fOa$^|$SwV7XgBOQ3KUv!TaL>OvFRJYKiOs}AfaXpas4Ts#=;Q+=(Yl2Dzbrrrvnr~H)zPO9iHe$G@4wraK7Z}n(TOMuF*>VuwS6Hi>{A9}R9G0;BgtcUko`K>Rn23kia_n_i8C3JNiN^SqDZ|MH>(VbAYolX^}eA}a6$279jpxgPIJml?|KF#vY4_oTBg4^EFeefi|s_8pl?CPr4eL;&WfNS*$eP#uKX+S+-MpoUeM=#d>S$7G1+n&u<l~hnCnyOo4g@L)h#UNqxyq&69*UkL>Xkf3TT&lz|Gb@6@DXTAv8c7Tuq(j|q&7e-=oAqwCb%Lh>q4Mvgx7i>fSP|OS`|lCQlacecmbq9{mZ;wPs|#nXdzoSm_#Xz2nu$P1g<3HXwBr=#hpRgi!Jo~L<s0fZ>hhN+kl6aZ9C1Wmh>P=km4yZC4$DKh%8a&=hH19bCpsYCAop?Dc+lA5V08twJnhVk%k#BvLR23p$;116iCo8DDH40Bt{X#Pl<>C?F@yQV>vbru#=a{Gpx&H3iFY0`>s4d>*=@SRTxOQ?s-h?hyVrI%6~b-bzpA#o+-NkgCln)WZGS#^uYIgt+R5qW(9Qn-wY-vGm9BSg_&bLI`e$Hb{Jwd9ReP@cV>)c08zWNF4o+0QPG|<SOD`F=#(*uaZ)3X0Rk-i==s-@ldF`630~qWSub&!qy3DBhcSvJ1Pb4^<b^m-0Zf#_aW>!bJdd^HwdVXD1Vs2S$syM8Ht^38QD*1i@f%o@q*Ign5qxm>ccMkrbp?b?)6WLdSo=rC%URgR~l?jExU7_<={IUiqHI~MWsZ@}G+?y^bkjuL=J{vaYN5U*iznIi_D+6$OSXS2peJcCsK!Ft>5V&7*GvD9Z^WRqk(Tx1ircV~QLC1@ZJoAQNxpU2i583+@)33UnWHNKL*^0V0=!(&%NCDYo)bwMWw$X}Y!&?6FFVb$_vai+E7kjfoLid`BStZ!k(gGQJc#!~yz><0nV>3Ohvk=Do8!C+&=RS97Q7!d`92@I(t9n3<*<#LcX`Rk1x9!({axs;nC8`i0iMIJ$%c$OJ?)t@h13CV!%$Mvng<`(4QWYquwE({)+(_{EI+U;EhNWI!?M&(7(ZB}cjR9MxR#+7|o3#(q`?iA&IL4p-e8&lgTdzU`-sTf0cvbC{dg1udc)kLJxNu9UVw+44WcXvVcky|AI+|~pqo=Fx`vz##;}bQs9o{WlH5CSCfKJzP>eT^jl8eL67HG9iaZEk`bV>GT@c2DSLs0P(2U&CAPPH=U^Fq1^U^8+Kr1f*_bxG#7q;5GAQD(GVFp>(BmvMR_$>UQjYQx`iIIT5*nqOsBC24>GQ2#$S%RgWp`!=6!CKvL$XtZF1+FT|Rc79(OAbuEK=x8kk)f^v}okuX~LnTxSzQ%{i_+zc#tr>h&DhK(PZMiLaf7dEdv7<+$fv$iH7?=8tu-U~`ZQf6}od@3jmt=W{z3Zn8&wuez1j@@GQb$g$6V4Iz8EqG>--}cVTECn%-kdgug3U0nV*Fmg<?b;fyW8NAu-9nHqw*Pl-uJW0)1UXqr85jAgn5{F^R{aKC0RZf;_M0HC!-Ttdc|j1q}HY_a>j>-z)peQ=y8z;xWpS(S1URg>T94nF&g2qbbO-LWBVu_;Pt8YdQQ*Gdn^fYdui$6%XViMkJ&bz%iL`JHDsl_{LbFa#*(?tuwNizS6gi^UE<^7sinS#ZN7z`oz>trdmbUCS*A?zaj%ZI9H$Lh@vEBE<;p-lwx1byYsoq^Kc-$o5I{WSwdW+6*B<S^9^A#a1<?GVWez9LA6q51suY7dOkSBV?P*XuxRVJA4F(+U_QGYK$$N<o%L}`|pO}C*9)-oe76W;sH8q+&qH*s6vj-`{W@K*>`%N?@<gX#nuyC2`hEPT>gz2&VTU4Ke5X-u=SB4C_N&nJQok;^2U~D;a?Kh*}kE>x!rkO3Lb|(ZjyPYDg5qi-;s^V>jQqCxtr~%%+>Q$XN+u>0A>nv6d7Q5Ii9-nW$n4^!Zf9$bLGU4CTs%(Ac>e*SFBDe%7puU>Ce2R0%P20rF_qz|0WT?AFcpj{j`v8-GNj_a&BeCu+o^&`2tM&K{l!$woS(fx0UpUpu_0NiK;T|3ucm&VmPG)FjiGnWiotX@)m9#bD@as~g<TL^a>CTHTot`5GzTURoGqD`Js=xr;$JYGRY0Tf_sCu5aZss$Yz{1UB)fXy{x_ui#<UgFGi<`ysVo6uQEUl+ywcLxzD1|`HY#61DeZ}pb^Y;jDgkRsbRyTQ5_?xU8<TkF&f;c1%gWLF9eWYs`87!HO*Q+v(Gco*}7qYBy?}?nMS3D?Br3UZ2s|V2bMHPZUT%S(!d&3Hb!-=n`>xuL?!FKOwsjg0v>sOjFIIr(S<6SSEDi_3jDSPuYwiuoY%NTfEKImoSZczAY9%)f>j-(3^*85KQ2yuPVR$V_ROLE%N=O5A_Pw4iuY&ETzeuYcnwX31=DcF0bze)WS5Alij*895mbZB=BKC%kn*IH|OWm_ZITOn!;_gV*E&&yeqg5vp!{Ik!fe6*G{_pi-!^Bwg4<EtO`KHI9ZxvwYZ9T<1#S-sT0o5)b>gsr~mDy_WIa}-%o?v4y?79%;ZmR#9xMm-MZr($GH2WVbPk9twFM|2Ng{@Dc!D;T+1=v;6{d%NP@uEvc`K`M8hPI-{GG%;@yOI*W0;Ozx+g=L$EM4hrCQ5B2vM`4$*EvT12<H!0Zk4)k^=x^ZcB>r&FK>H?9oNBvu%<Oe^V7;>a=?5=ali9L8qQj+*A7y~HPxW>sK_^qaLobKkH=GQ+m9C5Hrv<ZGb)j~5FEh!YoomZU3x|Z+iE@01@W48G>mP&c7~aymjSC$`RdrufqRpU*k9Kd^Z+Ytn&?xr9QgEo{Zk_GT7y9?FRJtAE<BEN@G2^$dX2^Q^ffHCLX+v|mZRi#AQeUoY@Xpovt#b4r_r6_QNYh$8EK6rqz9c8vY*aaSg25v~8pu0N5bQl;=0}(q_sDYoH@B&`k7>)IV20!#5)lXdr8z1^)52uUl)kn$-8?&eX7yUHLHPhNqesGL-bdK)2qwNzWQFADrNS4WQf=AZ^f{z1A%PsWg2SS{`>i>5Q2vOB)v~@FsjVJkX8UXR)HP<n_M+ZPxUwFUkd~J3Xo#MIBJ9?t8(^`*>)4o>+ws@rmED~)ZY@eTd=_yAAg;TF)!O`m0Euh-M3-e!8cd>UGKX=fcCXD>J7&`DYu`aQd>|lJa;;w?Ld|Ck+uc}8psU2B%Pn_k9#rv%tDKf46E|zM%0-u;4*T3yi5q@$sW5G46<NC&)_HLIZX_qAYoq5sW*{$M3zT@@8yJbSdwBGMxw0j%_y>!f<VvHtt?drks;S&u8MWN@w!SadI7mZ56GKkSknN)WZ>4XK+-^tn$pxms?~_5??XdUPx<ieGt0?oD9~y^&!uyj>%W9^tT@WvN;81ls#4TT(Dy{7zKGfP+1>@>ktvxhL?=)!GY`zN@|J=VXL2~4w(s<X(6{aG$AC)IX%ofNtQznu=IlV_?uZ#H};?YG@KI}eEqt3LdPo0{f;cQhgdj61DJwM&jliN<OC^Mb{Wm@lvay57h&yBub{53y9(a#Shj3(wKLmvbDY2MnmN8PQ`Cn^R@?v3k~^{9gCJp`$Lf5Gx}I(BoiUoS^#M3tIdpG;PT13bhb85M<vJm%*e-!ZMm!V!>*oUfmMiaTyLa3J##Zt)Crsl&10Ak(V%epa(JyJ59LsvjeHb}E<nnb9vNySuS&29xWl!&eC$A@1pr4qi*^=Niv>g8Y4mr8?K)!FwWIzI+3$*jWF@l)Vs*3RW9ZW9bRRT>#YWXOxzc#?BahBO@Dw67$h_RUln@OzN!a2VrqRibc~YKRLA?#9qV@dm1X!!fD+t4{N3Au-=uH8}+`hxS`m4-Ov3V*Zy$Z&8SiV&Y?a(7ei7AT6W+P?@<u{QBMC;${t(T);q@ew!A7eUd$jLNu1%&gbKKpTmhr`{Cn}kcb1d;j}+26P5bhCgR^qeJcy)_(CXrCjHCS!k$NlZiPCkoE@}ZW2=AA5HO=!=jpo#)+L-UW^>c<!mk^`_NpByWKlz@%8o3@m65|v72{I^V5Uxljx1(}Wz^;@~ZKbkXIrk^4nCnL(aO~peLjoIbvY6A~A=VKMk*NAxong5%f0)^dHI|egS3{V{#YYskQILVif+0}`n|0{o4jJ4Par#IwK*Oa3HIU2QVmP7dhGI(zar82LIWOTRbUC!|&q~PSGW)B6&jCFRBQdx_Kf`A2*{_c)hagN2DOzU-Ecv>{5DGW@FMj`;xXr;<9RE;mnOF~y0&KqfmvyFE&R~G-cX@5I=b`oPF0aDIoZ{v?=gV&ddA!zA7&8pFQQ?5{=#Jn;J`>f|h9bA365sh2Hizofio8&Kyz;&XPl)v@%UjY{^y-;yFHD#LenD4bzrGrX1Pr7y^NZNGw17P2*OamU#&Z1ulTs^pY-?6q<D+!7W*t_juONr@=k3+b%A$~|zLI@w;AH0hVS=-c%M`77^?Z5ci=FN#;A7S#v{*kSyKH!`>ceJO8+9Yz<jK}yd0787&^$~-R`=sb*z8gKT1Ktza=s7ziT;sR5(yhk44$=5UJo{W30+1GiWVRwAP@SYx}(8t&n|4Gw~Cf8vtXEGzQAW!>>MI6DHP&4Ul}7C{6ppMxp!Y=`?7Y}R@@q;Zg7&qHNbKn8}8{4#*4A`9X3K%v+jpnq`siXE4UMn!k;8CR@XJqQKqG}D2uz$e7u7Yyu;_N03G4`I^Le_a|c!bRy;;lQFxmT(R@}8-Y8M=?F)Wn^)DWRUYDb=qzNV;%qmlKyxpMJ_Pt|_kfyg)?^pjC=a=41ShsRMDELpJo6J@1M?qN~g^N8m>$AISWp^8345*aowr=#g1ZEF!-!5ATb%)GYD;{jK-znMHIrdzznx>)vHDL8vgjorEPW$@3N$do3@RlBz{&n>bFmHM`*?DD;PieScW5GIb@Sf}LI!;z;Y72b^C!OCnWSQ3gf}CN2jIqf!q}_4?n%<nK10gDN$a4Oum4i~lk7=wtt$U7@L#-LPlBcDk3X$c{WhEXu{nsd$*G~>>s`yX#RVsth@;zn-rXtYnVEyRnNB@>UaXL9umGQoUH0ymK@2J@?<7-Vv!Dj50OXCVSUAgS+zBVzfDB#CwA`yGkCLh@df)(YwK}^qG&G%*<Ze1IHY~EVEY(#r^HgpeKG}TqLm6ByAr10xyY?$jEHLOHXw~o4{DnTAb;KC)>N&`!H&@aZ_c9Y7zK^<$A(ziKk3^MxIo5-d6Ow84EQ@TB%@ob)dl{%~Quq%r{?MzwQ!LeIG;mm>)h&JwOsS+K##J15oonf>rEKBF@+8qg9xJ|iO?yqaF6}r6FNuFz@&3MaSzW;1B4~$Q<WW>#Nd2xF6>6s7nMBd%~0*{E=Jp^x1;yw%+w+AYlSVPcRAG*&nB&B6&a@4akeTsMe<8PJSk&4=mhI%{PZI7_Ci$4S0_nH;%FPV_hDCa>;^`BBU*f+f)08Vuq`U2wyzn+dp^dO{aCPmcXsxA!fO(1U(gHrvuy5#O_7PFOcFuSuJsP$?BpCDR`85d2%5V!rpD@|!PSo^KB@@!JtrdYV0^_BzRWS?Ri54y{L$L@I;m7Dvq_qL0Lc0M@{Ub{N`mzak|=Q*D*A?mU#DL0xB5f|PWq_A?-_G~u?Y<nCE!)(*XdkVE|l|xn){{@?n{d=Qy>^pjgoxe}(Xj6>Ept0N!^+&Z@(5^jqLP$t<oz@*1TFh<*9IRhH(oOW2A-DU6p{rf7y`|~0O$BPV8@%wRYi3CX;)|LhK{dko7ev3wwnulp4bf`b_|!VPLp9xkLio;0Q+G$r>HXn+*&k$q(pN@0O1K5{<$j%hf1t=selyJ;H>|3U&k-cxzaSw<Rc7G^#-hwC=ySE`pJD7m-igbTYbdwd3RdCq9jP@x&NLp0xP~ZQFH5u15wnTLaXY<)<yFtFtT(+s=>(80$`}1-d!Oj=k1#ly_-E&})4R&J2)nHJ3v^h_=PZ|`%Vb=-c7!_yEE)nZTzA|fb!|eO&67A6^zi(1npwYWdw#bE-K?@W?>cijE!V2IzjYbD@9kfCEs91Thzs99T&ri;Ya;LYF16qGSKYi=g@!Zq)muw)Kk^V|F0@d5*HFD)xT?3PE!Em+7ynu*N27r*a361vfW-g;HjRh6+chDfdYSZ4vUR=<^SY$VzZgpLh^%%zK}(pPuVJr|0?i}H$jLD=={+a)m+HF*QO@V@U4FucnQ=v-Go^GsqvUeP_5mi8KH{M!5FN3D=}~L7G4zMnVY;g}oaPj|cv(lKL9(N(YybX_>%Kiz_j0|)$9v~zC*GXrFQ!_wP;jJNp7qY3dLrK$pmivJL>qb0gBCbncQGZCddNMMt--kZFXav!EriB&0}F@Y9eKltcGPi_p)%^+ZuO5Y*B0~8vR!l|(D!Be&K8TKRm0+Q>BPKyljU*9`h-q=+U#BBk(6lpD5O8?xPKjW>_tL!ZT~MSr`--&7^xV@Fdd4~@7Zd-B}wPMeUTM1LJ7-?$P>g@I}M3Sq#;ru{{0PNP1bLL*uy?d!;XB3Wx$$l_(*%5h27Z!XV_+b48-RV5s>ojDC#cT*An{iJWjJ(aUWmM<3#ytSV+K?+mIsVqB;L%ng6z(Gv~zLCOA`4c*hOd+o6Bl<`-u7F-szN9E@pAPbaV~Ru;A0@(ztsFVYw$>*z%^D#!$JpQo#KKnhZ7taJjYap&>_7x=HZu&Pe|p%9La9ow6<l1lUf+7m?Um-2a6yzL*vQ#8i&&|6~A<b6Tgew&>4^R}^Bbu6yte+71zs?-d3GjcHkyh`QfH~SfLCQjp9q_{gn5OJDm0Q1?)Q#I(3t0!PEQGX;kt6-<(hwtyPkEfPr>DJ#5#xKaGQjyTrlcap7q2#*_c2!r860hnk#i;&Uo>G2qM&uE6yG##qQ7jM~XKK#5fiFc*cd(?gtyU2O@!*$whALCe@R&!#BLh8DX{;H`Iv|_aX!oc%7Lk^we};4Xt7tGkk!=lw>M^}ZD4b)?7*ZNlH}2ENL|bZZ1zFI&#6kTg(v2&phc~?0N=*jrBUuH)YN><>JLP<z<d&j2Gwt&8T=Up+${d}+lvA)j|3~Q<3143m@`YAWD7KmPrJt+P=lM)l-Qn3^tqG-5ucE8LDsH+TFIM`;Mz1;Wv9%-Y?t{;6+?Vn@J6DIiIJ0tom73)vKaxKDwiK?P!KwcAi}@*9Mt?<{toc4Gp^*i)8;9*?g;Bi~P}%zH{$x`0R&Azk;d8jAFFSFqN0L~u{Us8e4GE#x#Lz3772Xf%_j{*YuGi~s+<y0y;L#xL#w9g=i!of^I)4SmmimNOToxXpr+^HVRIhLF!YVP9>(NznO@Wrv!@6|0LA5u#yr73QWT+`kANIBTDQD)NSks)LRE~w;0x9sbnq#-!8_D$oMuj%u=r!K_Wt5ltob9i$C4j(9WdC$uY}Z<6T4{7!)Hs(W*L(gj&TZ3FWGME49<wao?mKh4iCjJ_%hC<D8pvA*Y-~|Sr2F@hLg#vQ?ALnU_<_EvLc_MnvNm^$sC)N1<;`encY2=QQrc7DGao+1j;UgZ=EF<CUg1O^tsAw9G&#2<0(@K&DW?E5VC#~<;?KU>UWnoL@VDVOJ*o$4w42ZQ_%U8RIs56#^6_4rCW~sUv{OSaPhpYYWNrAOBw#~tX(60YetGqNO>C^9+Sqt@clcp8hiqV5zfy&7W7MRb^I8v}eCqW2YbEXcmy)PIP>=0zVHW6>6CnHFoF|LauIoOllx%!8Tx_qpnvP9$15cz$$wALcwZ|-k<d6%1pauG-qd0iDp1%8gR^eLhrQ!rH4W|Ph_fIlV6~3n60gY59oV1MF?TR}I%Nm9Sv+d#)&W#R+$AzZvZj25WZ2>aNZPXrBXTk1P8yuG=IN}h_u~4(p#a{YpdpRDO_x-SzIDFV?EAp606}iD@f8%=v?l!#+Rf!*Rwer*5qV5Nq{(TPrIbE*q0}d_j8S^$96I>C*DHbTxdeMRJZK*)_4Nvv0(-5}Ma(jk}2)j+!EqEq5zB5QNvj0wD2D=2}aK3Z(_kIilM7?F`pj-)=`(*pK<eatQ%5Z&Ly_)zk+ALd(K?@*v3jVs&*BX_49}QMVa8z2a;*;lf?ZvDSVlKZ}57&=FzdUeuVHqYzR*Dl0?Rjkw-n+zQc4<e~8ouBk(rP|_8b6y%JBOnO6&`=z!qqMuDSb}y`mjswW#0Vic|X0n%lvfGaWNbhYam|g6z?qHAC<bl;Px~p_T7a1CL=|YQ^-5V0NmlYM!cgM?>8IXbcK6zqwwN6oO0La^hNMk&Tm>t2de2jME-(^8!~{_3sQAXb!_o|Hth?10#o}S<LMZu%A=FdKm2UC9*2LNQ;R5hmhI!^0-EH<-krRy-#g)&%Y8na1h6yN!&*`qBlo8Db>+}5^P;~wpbs8<)oM*=x;76lm5(Vz2B|n}Zy%FWFnSi_QexX$4fINjQ8csOr;0hqXIYe(j%^*VLk;vj@H$w^Z6l}La$v9Ves8+-%<pWo7?;Tr-~9-sdc5hFQ?in2i_EtsNR_<N`dR}iX%q-s_b~w7M+Q!6g>cdFsVo4%XXW1Xr`3Xf3=4U@<HHcG;QDu8am_5v)e)NyjYFIUdaaI8j0tr@^s|B}0<l=R7)Rl4y8Q0Z92K=X1s_C4Qou+V10(k~Z8&4Ka;z?6I~eR67A)1BBa#Zsz`j-4#~gPzv^0T1ywxtP1w=rX6Xd=K;e5dw45_>VB7Ag}i!d_W(bCv${+f#Y@%91!#JoY8+Q3`x;L#|aIV^LyIeo+qijq3(^bb(G=ybK|#Ndf0SsG&-C}xR<-D?p6F1*b?XAE8=dfXj)Vo%Mb4y#N9v3VEtAK~Hrm8Gfmq3>6XBWA~iP_ay#g~C$N>4({QlqoeHP0APGKH=ipd~eSPDr)SQqcH_Yt0~T&>zgQ;>pd5Vha)Q{lsB3TuCnG7!a;EZSsAuV`s&YS^~0~*+@>}eJ<<V<!4&Xpjq1KRUnQ%4=^~Ja8dG$}U-R`<^#s2$jm)!763sQqZn!rs<y>jRf)B1A<_6MsRhBqxYLycBI2=3<kUPDE4O+&toK`MxT<ryF#f%L*6r1z&)1HQnm102i+yZ@cEeC&7-YV_&=aFP3e9t;oV~<PP4PtJWV{bfnj;;l+Zm88bR9YknT}qQ*gjFE=8AxINV%TM(WD<E@?ujbwu@z)MA49tIDSBsXc$ef?f8FH7@NetZ<KvQ9>i{;;8@QKM#b0TgI-WR~DZVG~;VwvtkD#L$+WZ{i<G!ytTlgxVZIqnYbiux=5|5z_Q*6E~pr7B@z#~`%naz%q4=Aj5Mtd;Q9_L3;_o0hd3R(MdSy+{z$!_%HD0&0n_cgfV*@#ct@+1;Xb?69@HupvHNw_L~?q5FAB+0FAcsU(_{*4^JOZ+8o-yU!20l|gbM75Kfa#ZthMwq+`q_vftWiw0TZpQM7Yq$O(1~*?FA6u<E>(j#&1&vo`mAkGqncbsyslSn1^ZBSO<eNp0d-~dbXYZV@zl)!x32gixa~iYEWwd_G;9Ldw%}O;Tc+uttGF6d_M!TP*iCy}tEPKdfonw={YsizI#e>r0R6ShF28B$Pr^!$Tj*8O_K9`}`;{|$Hd5#40U_kGpo*0x1b}_EEDb1f|^ah`=NiF<$FIOh|@umQf8#-a_WAy9Se5lajD9JheK-G3(r!U?>`obKDhELxU=F|Aapc}x`7vjhYt~bkD|Nd^lVREk5niXtYjQU{P8_S^-tAI7PTj(EE8nn9nb<){Kp0`5=uOR``r|Y#co9-J&?EH)%20#>Q`-C>#28&ju-qsi;><kJNUH8eFy)2(6t@MUT6X9|iN~^BED_QOE0Q&KGPY%2O`Ok5&3F>LRGIf8(&vM|jpvFzeUmH>)n^RTMkgGZB#$f9lq)&g7?iptrMu(k!SLh$)NH&Q_izR$|yDyphPv!t)_$qgr4v|i^(tP=K1-)Uzj~>U-yx_d)B2tT?SBSfMDM9M(96yxCGJ5?{^XIcViBs8Emb}u%)eOGbg1`O;BlhQN?xixq4T0fixKxyL+mGY%Zu$%HkgCB!md_jYpBZ63G_~t!KYR*RSLSO>!Uycl2#<+gQ`~Nsss*=``&(ZwUlXQr%bh<@7HHI!;P50K^OW%beXvN1P7n<F9iJf9$7>|Yvp0&1Xxxzw8-HPpXD6rn5zOI`yAvUOD9uM$<&2pFCdTzbYY&99G%5okNul5RZnzeVd++9?+XuQ&JnIf4pwSRuI9-1(a=N567Ld7FabJF@>gl*UPCKSp-QEyu^}uU8rXR)VOx*ke)GhOs+R<u=m&?#%6PM=m58O2?guE@X;p*VZr8m&16V5lBBehIs^E7%;-zMUG^ZTI3#P#kTN}Xw1nGbvALrRE;ZhIP<L{wmhJ<04CZPwwUWXUj7+A&@59@mGwjEqe27+1Wz+G#D#eGI?YmH4T&fWOm~JlM%J7_&g0<lW?G?Ds#b>kNA#bGS1RHdv!bj1JMe?MHRuf^_95W@k+y_BTpi($sAUvX1MeZFhwXBX-v=1(j+O2RpfRxRJ-9DN)@*_=5|E=H6i?nN*Pzx7>WDl`cfX+JIX07DC(M7t`60c|S*k58>pki8B=>&S+ZRV*B$`M{W-=6bGH5rf=H0*qW$SdwykEsCF^3ZKY0^vS;h-L6E9bbG$o_Zk|?H+_)Iz>)o6}hFv9l&@=nNu)8$`HMj3<5Z2D|s_%6N(C*ZmwXf|@27m47kQ!UB%4Taef|tk(+*@R2lm%y2a9%~UW&R)7V<)DL_eyDNpj_wo<wo@6+3ML_&-3%`j7;dobra5+PQvW><9hgbuO#dWwt^$P=+RGY-}}Sow0CQjy4YnP5>_2SB@GxtAe6VS^ozRasv)cSHY|1eJCK}()&OCMYFpY>xf>Wq?Run7!44f4cyt`LTZ{Fg(ShIWI&9c@SKPgfA(}pX_iyFn{BlKfhYSxuI_{220+|KUajpL^1&TW@H#%TL6D#Bu?#7cHaj1kd(~2{D@c7)uXw%v7I*!6A1m|aS20m|7cV9^sP(2(zJ?xQB^a%dza?!*zP$5LRZU!r0^X$rUINWg!6nMtgA$(7S=Coo>TChOzivjaIa*tZwcbEID<JIw|K1qVuI&=I=WV-%s_jGnDzuiA(=2W@_-PYAKI~Dh9gz6rqKycq~Yn-?^$;a8KXU6*Q{8<#+=<zvg1SE1vgD(0@2vMp@o#=gL*&5Q<#=vgA0|mfdfM}uH8dyz{VTdk+<;`5n$9KkIeak{_e<_1r*s0&oU3Ilup}Jcex|8Ml_q<Q~OLH!G6Yo%}?VEl-I*^Qd#-8szXu?jl{Q`zp{%bAWk!ow>GKh^b{?50wFWTe_hNbMG2W-DM_pPh#>!8)|*SY>;o8i@K)mqc3wK$g~$V#h3OE$|%X1Be&(G(Vk!IOKAwdRe!md>Vs%@Swy=K{)`@{m3?_QNUp*0db7!8a--B7Z_hwkd1IBplb5>Y#RePtLnXY5!p^?-EFB7oacUv-$KR!RZD$GK9)Syd<?c!K*%hUz^iszhYJ(cN2Uc%_Z*$?;Yb{MD4wEo87Eb!-8W$@@FvbO<TpF(N)&kz839Adfbiab(*NBL&XB^JM!^vFeG2m*uU(|xHZ14x<O;zXYn*7`PThov2?Z-2Y<Fy^JzAFyv{$~y(qVw`nbnh{;{m??{PQ}+Y3Wa4$IWOEeU=aOWV)BjOip~M#q;mN2+GD9K!-F$BAa=-{4Yg^bzefnIe4ovgrk5qF-f?=rfQxp}X{YQ0a2%P+AZ^OS6P3(Kn8rDY04xu+I?p@?f6R#3%LlJEGBnRwt6N9O99CgCngKX0XqqyNUj_XE^va>*S2^1Ejv&nhP6UH8u7w5cy=Py#2fWCT4@VP43%ab`glm1VhD9B{~*nzBEVBK<$O6jlclRQyo65^U;g?22B~v*%l6}{wX@tnqW@X*1`KY;7Nm3LU?p0!@&ZW4N;=N=C*NwN_yXlImw)Q2{ntm<_SK7{<(Hpp1P_Ck(a6_--TDQ1t;e)7d)Hw0O)T|N7ZAga+L6{C(g*zE_b$xXN8K}tMAW)+3keRAH#bwRFsJU09jsWHv4#vXmc)0!q07SyUouyh;E<RrJkLz@f*LPty+~Anfu&?k;SW*8-K33klZwQ<d-`6<xdQm)hY~yoK?ALy_XI@2hF{PypV2{`Ggg<Sk<rT*vuUMHe0TKAKq){!Tt13Ol__X-&pn(Pba{H^h<ZKJbhdMnwXjNH_s#af7g+KSM%AJDu*C&yFkH@_TytC<+0J&DaH%JW4W9ez#-lA7s)3TkVdPMbn|m_O)rS|nMQ!tL5h{Adyv>lPg!4lyzzzgjrjOr)3>y2j`}C3|9D_^7YDMtlTG#oExRvm$3JPLQjk!1q!E@!cc>t=J6DX3-$QT3rmb?YxaY6S-7;S8S}RM-MgDf23*1{^$vN9|UqZ81(M|p=v~$?4g?d{B;I#DvjdUX4iHYAZdilzRL^!;Teu6AoFLcsINKi_r#@_k)?$+(r3~Jrx5DfQLWfs)9JX&XSIOy~)Mu}_B!x5+qE8=;n6tl{U4bE4eM-bzrX+xB&wS;lrp+Ni<%@~B%Q9ivH&w)krAHWy7)kA3%413SN2GQcczhs>~FaxL;cH6vj?8Addd?=UVEiL*sRrXwWgi1TJ(_0VR!;Oq5cc~67IDfou;}s>k9mk|wag)=lF^+p4@Z3ydR$Yj(H0o3#)c#WsbjkoUUkGquS8lun$MP^gY59@(lDF*HF`vzz*I_+=W^Uf3Cg;IUHgBS7V51S+Rc0lu?)$@-Vc)rb_(eDBIIUiWG}SVu$2o6IxPF}qKGd_wnm|piGKWC&jc|YuWZT~P)ZkAP#e*g}rN^eEC#`%?I~-BOmTFUXIWOc0!qU=Xjl0XHFt_Wsr02ue3pX87@h=Tp;wJqtmgh1h8ex(G*VVjlRy@4bNpCy7**;>&@-XPY>fb=_HlGeoSE;oMfAG#_-?bv^6N6Ii@-DtJm}cF2JIB@F=xu6VOOnu-JWup$WJjbFSiQ}7PpOS^Q>;~0=+L-yY0jJt9)ttFLfrWjS8ocAXh+;!yW*6(-7v-|`*Ap}=m`{m#LjkjeE!8Pa3SqfrP*Md-j`HRzV!>Uq~re)b!|=O@=^4&G^vTkj1HnHQiSACC~_esI-qoL%HgxW{uht;{+XWAuf5mWd+oJnJ#)2vB8ax=T(#tuw42LAc_|OI-<VY|9PaP;{l^Sb<EnnL5TJz_->3KTwa{|2IXMjJr&nmk@5`|((8mekr>4+E?_n%4_)Kczeuqs*HN1Vn9k$*I<{N|KDu+(LwVKxQtidvWWqWh)7T92fXHPC$X3lm3IM=k+#Bg=`sOI<XDBXkMd_BK*uZ#9YnZHT+^B%u<#ZN|7K|ZXt(@C0}3VWaiNBAgEV%Apj;1t+*<l%K5)5`VFN12hjvD-Ynq+_~XK050~1`Ye;olEHPLP{sq3w80F86+I8eKB7){VDMTvMW%k)DGk3RCM;WLAq|yE5Pk>*jZ2?7U*aS5iNql>zwG=2$uf(E!3qmF*8ti9I(5It))KL^PDv9z0dcEiKuS9o@P23d*{7Qu#*eMZDwW0qJksVOiW0!cSV<a0&a*;e=dyO9SPj=Yw=m<&Pc@3<5sw8BjGaNbHj~tKstTtyxp~RdgT^}ABexrcf-%7-}$8SCaZP_t#!IDv!}eH)#Aca&x7j|FJ4BCpTFX9hn3Eag5Bn%9#~2<pxxcqjer@n_zf!sy}pM(K5BZB%3|7j72{>ct2euBp|3TaguS89nkPI>c|K+)0ot<yvc8=et8~Hi9c+c2C&N>v^~dpLtUiA|sB10aS|Iv*=-}P21Cx~#@9g^JabbKXX}p2dk^9Pc9j9AMNgEj72gTb=&qbd3CQFd<(NC!mR-iNat{O<6Z))0hh?RifXa>=WzL(GBN?^?vmbfFDw`*fqS;~#x+B=M@hwf8^_U(+5r#)*L*_+08ola*9112}U_Tq9gb$NOv@=0!dsopLu*d#A1Z5E6)0RXGa)w}+|@tAQhC9VQRyZPzP3_AgO(FqBi99Pe?7Uf;fA1Cr{HfBHgTq!owYJoH3H&bpk`p<1>KaNSg&OP;7D}<06|0ac(yC*8G9NEVH^fw^Hl;-UXGh6hY;NV_DP64ro53?_0A<s&+H9unoN7*mP!K0}${NWI`QPlSi0omNnqULIIpJ(?9)G#;e#ifOv>e%i>iVqn}CgTx^U8Y>ENdB>=&U#B?)q}F$E$`69V?W_>@wgmjH`L`v<2bXF=A+n<28IA2E3EH(otr#u^@2RdE42jmiM(6=yl?X^S8Xlm_p@4Bf+g}U>JTC7Vk4bBN`KP|_<h)mXB!rrgjxf96~rc44fm_<X_FjOV$dz0#OzlZAD(jmQhlS|M0;%zP4r;e6k3bbH#4;hU0X{f4}s(3aE6duEO-vd_<}OK+pYlwwQ;nfF*%{L0~;VRk6sRV6ad#P-hZzH?O<k=l*I49^;ok1`LK^_QDfmo<83#g=QlB&UH$}1bE(*B>r4q81+BsA+)n5FEn8_3%V|ei560W#_KF%S3FT*NyDMW~Bp05B^ka%Wdg`WgIiZy+I_@ah-<D&&P-8kfCznglAuHMSt$jwsd#6_ajVUe&HEmq%#tx*MD;Jt;Eh3x?56lB-_P#%+HYHvM>!vK<o6E6ZJ03@iB^!j*J+w7vrQY<4MBRL#NH&T!7VTUnRe8}cPki=J+_~~QVR&9+6~H{sdnM@5pDB#pSp_>fhuJhOSlTQrp_9fXsV4`TbL#?rMKlQ~6R}EoAI?;A^V+o5P+7l0wQjg2)NcS@zVsjlURQ&n#a-d&D9;_7_mEX-!MH`_wL7AJ^>292>+3Gu8nfc~z%vi}g%s8dvYA#?UNG1}<C)AaZXUl@9e7ck=&JCtn=g67@&lXyG@X2G1?>cv6cya^-Q8B<cOt^U_L-yeM5e?8^_L*gqe$SH9Dq7*!d_Xu-_FN(SP%Udw<b9on0_`R1A5_G>!;Zw!-Ed-{HNs?Mg_Tawev^AJ{HzsbSo9s)UsrH(*nAaj0?v{yABgIEoSo4=L<8-eF@t%%~OE<W;+q?C~r{Km@EOy3+DDQ5?6Cc!)<imo=a+d8ld4WN9qf<v?8Q&#$m>qX2)bQn%P>uM7_~Q2n@`%Z4+;_p4R3Z2YM@19#XAhkCgL8EJc68@bhU<g8P}yUIW5HFrPVp;{^N^=jD6E2ileGx8~z%n!pERwF^mkbN6fWda|J}Qgz>N*mFV{x{p=ic0xY%=Nzzu&-%c2BgKk=-T1u{f<Hy<%wnAbrxJMov@lO=i5@?i$E|<ASooq}=oE=#9~+&I-0S>`zM<t_moHx~xO3aL^Wwl6R6<o$!9YpF4$c7&p8&3}_P~T!1c!Xg-S$Xp{c)ED0j?iQiZUzhCAJ&eV}DW7z-cMJ))R6Evn=m)Y9}rt;@T#^%_35E3U(k*VIDryuUqt}8(%<HQ1+kTy*OzH<hpxBafe(tioHsMt-lV%zi$+FQ;@ISm1)l&PSVG`{E6BkPB|`;&W4+`T^>Bw4bRwZH~M6|uV|Y0rFg#Dv*t8Th0i+VAJIjCcuwK+vqWS_r@vMAJJsxir73tFJ*Hqh!Hb7;y`?y}f&;JO6m;}NZ4EzXvYeKbq<rKb$-4jU8?)pn6qk*WA~n3^%94sa#rNCSOQV^h00-9laY6V$;bxSP+l`Qq*y2ZQ9}QXpCXf2^JPwGXzN$az$gFkk1*Tszz|@WXf+J_%!offcs`Yjo4`59Nk<R=9f6J}9xR*K57<A|A^V}N`qOLvMV8y?k0yAmS?fbg94TnAPJ~=0zX{|GMuo`i*(k_zHkx8LqbqkyMmF@VO%8d<gEoKVMH}C#q^Qrq*pE!}?^4=Azo!e=RwQtI$mNpOWPQFi;C$977QCAZKgkau34Fw=fqoF5~i+TqtkalVO1`eeY3Vr-bU6#G`?F*a7(5Tv?CbeRH%%Mu7$Qan>&~Ty8$}(rqPgOYxybl%x<Ob8SalJ2ihc%XJbQ#fX(*gg)E$^rkk*gB<(2j7S$d;XD<Maqm>szY}$$9q<jNy~Wpr=K#75aSO*Fcl}%=VNU5BbOz;i@odGl!tkOBy%M6{)hCKp$lMnQvMGaYoClw5<P`9}f2H(ChG4G$Q@E7<ara*ozZCML(VA(DxZv!SF`=csOFnqPQAb)OPZb@3WD~k9vjGL_6@!>BHGyuTgvDvtX?{kyn~j>aGE}HY7a6)GmJveX%_Z^mw!ISMHjL^iJPHce`%hPX<F_->Z5l&|VkYh4VQ^SMP$3Q>q_7zl_<kx63h|(*|k<YJg)+8Ck{5<ld9QW1)$)2jN~C-ulu{v<%M+yjM5P5=&BFIQ=W%xk9%dy;$yX`m(-?<w#e!y*b;_b9YzVLmI_+M_XV;$cGn3z}8fL*F6>>1#N325J&6l7(lt~Jy(4X*bwchSctC(QI#i4#cu`Ag5SE?zYRzRz1b%R2~F0a?Cv}Hy!+(bYJ2(IO%t$$;6&JNYAl)R+bLI?@_r1CCDMnV*xfw(RX}<46!q1ObDf|atF%V0Ep!5v%*t_nVTkvJpNoH^rx8H69@HZ)Z5z7XVK23t1Fh=VrV8rbO(1ovbAZ<K>2kKnJK3<&iEoYW^!>$O*=%$+5DW%yc|FgcJ$q|O*<#hNN{yoTMDLT;MItR&zOMg-Wd(M*cuk+`j`mZ%u(9X&{=3uB@k(FYV#whvH6N3~Wq$7oo^e=pgg2O!Sb=FXaLps<?U5);bRzUiYMX3rTW$dp3{{zbX~Bv^DY}&o`PN@Or}m_uIlE~#?&GEE>2BwiNlax;h4f~!1~027@}<Nzz-DH<qUFy($A*>X>E$|0Ft_`RIy&2Rzb&+RLa&ODF}C#Si9wkGbDnyGVJqmZ=|9CrdrsY5Gdw@f{FR5(@hkIdviaPui;6t{{vzf1;l+~?ms|A=01n_%<T-PpcV;y9Wh7I>M6)(8t9yDZ^<oJ0Xkg%)9Pc#9uy%1IbQ}Dfn?iGlVA=vb2VK#*ypMILzq|qERk8d|`sC<P_-RNjE-t;Up1HGLml&?;l_z}2Z!O<4wZZB5D!J3QiyitK7CwxU+=^xPbdI(i?lPBF?T<Y4o}iITR?PRXxuyH2NGqv<NK7skrX#-6>8zvO-|WYOl<T$1=+TBLpmB}WAt$NRi2uAsL*(Qy<z1UvWKQ3}q_ti@gjoff442-V!rMqEO`~+KkGIkJ22KXhI=klA)l{E{(xW_^KS)|q1Ot2g<?p`Rq~sDzY3s+Qq^jfkRmL9F{61%H2}qm=8#S2|g9*MycYD{l=NI|BYDg1?#ya}tOEerD=)rbV2CPdRUS1raS_NC(HMhoZV@`w;y<7eplWu1ZotQealg&U<cjLg#Yk&FZx4V%LfjgGJTkrZje3J1z>OsPVW&ID#?~>N8+)FB_>1L$WYm-3um6b~3(m!tsgO{+O2g5${9^72`=-nZSS#Us##_%H<9PEcL%?ER)(u%FQgQZdB{M(ac-EUxFJQ-RUy<)F?XCmK$3Wdk&Biw4oW%Is6hPm=-Mf6=7#?nbD_}LrYC(TL9-7KWqJ_8Nn(|<mK>P#H$aE-Z#M{P01sw{fS?=G?abKSCf8(pjq?B<>`0*y?C*hX#(>%68XE3Uq(Aw#hnnra%A8U<~2g)9Bxd6;b;$G06z7i7XP$KaT(qS+X2y2M$@R?LHB)QE$+q@8lF?l$>rMh5gQ@Hww6*Wro+Hn3T?oHmxhAPf<%wrNns;>oRU1FW`XM8ot(ujl%St<EmI$5B0Bl|HF8nv!7#zkX^cNcb|S!e~+H9`eRLbUsO=i{EUmtjdb`KA|fr{UQ(<x?Ps+`@~ES)uy9~E%9T)<TB2HInem$w6^Jfss6Dbm?_(ps6|nI?v*}tv<09qb^hL}I8i@MxVeM#1%T4*J1-XZ1O3tW7wWX#&;ZS@5!%6!EQ_&>(b5u2J^fBpvGfqrG&~<r?8ASbh+p;5B*<d|Rq`yXLjc`Q9X^%)O+$C7kfE%(kz>5oYGqB1mSobH7y~Ob$Ms@sbbqi9jO{q?l6a_fU00jl>SwgiOe;Og2h7J%Z#IXUYkN-}!Qv<*h5#~t+D%k$+$KziaFyZ*VwV&6(`ubk22zr%PzJKz^2J@2`*BnAG_Po+_Qy7uyXKT^CJ&is|EI=414mZ=2|B`a17sh)Pg)Op`>F*kn>bgrFEqaOC(r<ol?CpdNUkskaO^4{ES;mC&30p(w!vquC9c(#r#}yI&NcmtCTpEw{B}k_%rd3FZHr31t6iaT^*m>SdePfuH-jln;zT+k`X1;mAD?0L1jo;IW4x~>lac$@ADE|of895sxizlQx-O(W&p<0J_!)|pni`0@x!BXs)8omIw9P{-@lGJjmOb#agWUd5A3zaxbMDP^g^O2VizyH@)N0IQWRd0%GdGUq>4_xe`>J|<8m~KVR@K@zQo^%+d~Yh=-~Hdl&ryh8mZ6t5GW@NQr&~x4)`&S?s0~o7o`lg_!uZfQI6($C+{(W-F+>gM&wft!i>g`Mt-trxZBL=9$b3<YvZ(=@Xjdn_{Q#j8E<RZdwksBjME}WhZ`2-ds-LaCd@(^Os;KI369dyeU-D|NG{7c<UWK^(6uT<eVRtxdNh2bq&#0}}rn}vJH$)S7b}b#N{@L7gWX9aHYjL_dX&+T3ONBmxbj4z3uplyPkG`Z$y?^@N>+=(Ny2(rK)}fwFM8~ri#F&iPOZ%y`f^Y!@ts(nXiad{?4j+jSa%q&U+K!Do;GkE2k-Mm{oTVL0)bzjs)}>%)_mr7;Bo$4+>FwWv2iR>zJ)b+XwKQpG*?7>6VzgE8(-c)+i?chU%|_)Fj@>HnFaaC{-VE>4?V5Uw4hs^T(=a}|tvO*Fm-|P`EgjoL_g5z$l-6HPhw1^{d(Wfe<;%&T-$<aD%^H0PtZs)rI>B|L)_6nL`Dj)AtW%U?Lu2tC)*3*dMWm~<T>r(Co~DJhfVNS!r<-A^FjOaO-Z_4>!}@gmq`yXGgGxKs^QZ@*cCl1%ARy$+v2LzqnHQfLFFR``fVBzQU1~FFJ0()gB?rM;IH(H658H_}fnA|+!}|{-{xUoJ`KE_)de~?TclGPm2ANrZu75b8-`$*%{7tVbg+{>1lM5!@s!dr07w|h(ZqHhuh=g*$i%a4*N~zm(Zsp#tV^?{%Q!G`3DZSY}+5HTk-w$IU)UB0F%R*VbTI&I33&Ec@*l8=1&hxyLw1@NrrE6<*8bGxhuohsna5J3FPRQS4M;H*t7eueU_HyaBDk|Y!8p05|&-v$UPPEUrNQ7G-dzjVW)ok<bn5lP3RFPR>7XPOfFc;mpnd2m|mi1WFZrP~{3B|l&T(m<CS6*GGHzKc^nye>J?DXmG)>;h?2`wW|L2dt@UQ2oH?yRZzY2P;BigE2Y+e0b5w<=2T(?hkWkJ?L=<MK!+)xvt+++uQ&Y^BC(F2&-A?>DMW9AvaM-!rXwPxbBgbpqAyXT$2<W-(fH{Eu%RvMI_-m{SxUzyvkPYS)QLmf5)ZVBr2iTn2VF-@C*0;gWNXeY?J>LM6br$O};)L-Ddneo}^*Owqvpob9B|+(16_-Obsl!qdHB_L9wou7!L4j*_KW1A}gP(9I`bUVmFST%(7w*<Z9eAs6@U^5_>L_64n!zKRMLs`q-|&oz%-LubcZgyufzRFVaW6%KJlBM2Lr#m`ZpzaLWru`-LI8Wzqvdw!J5xMmD!s=5O7xf<Ku%j>fIvyZm#pxfmXkr_7YZga`X?s^1uOO58FWm;!>)hdh%p|CGp4E}|8i))rWkyalbIy`Vtd&JmWcTSD*qjKroY&M{={KR=DVcCQYlLNK!t}2ZGifo(rWd*GJ-BN|VHp(v;kmP}Hb|yt*;R0{@)kk`*ykxdu3&tjbTb@hu%Wg}ZRiNia35U;gFwVxov2ma_Bdm042-=DlxrcCR7ACJs?90&KyIReIYX3aNFhDDgdCa_NVRti$NBl}?SAp|MMc&QvzP6t42(ShaM>{EW`a66xl8)Z3b(`tzc>0+=j1a)KZoI*!<2}?eR``_b9v{|zO^wyR##Pi{+Imc)X&<Ar_iP5XGLD*1E9{AO8eHahQjOqk>{`{f;<#Jlz7*jKETm`Df;|tPY#Hy?=k;F6Y?4_%SKDFd0k>Jco_wks&OEn$K=}={F<;JFu;nw>y2RjduGQzWNvn7swW37H(Md!VPu?E+Uc{pt5;Cl6rG{8SfjsK`^VERLwD*5~x0H%)q_=(-n9Qw2)UXou`B^FH;U(fePfk4v6_lG`Ysn?BbW0OUB9Jp)vWrfjlCycYA<v_CiBYMk7esJt2#*dt1<K;4zqaB4c<SNbAY^px2F0nSL&Hwm*pN-CeW+{0YSB^&VtGyT{^e9zu483+)tBaywj3D@hwssB#9#D$XKjj4bF{E)5)dlQWAF6fhu8JR9_{MkPR?d8Dh=0veOFGM*-W<gs@tiNvd%}?SDeCq=#Q$(c4W@I_w#7o+SmNi!<AhR>OFA&xS?5Hm$swL@)lQeXFL?8Yy1|E+Y%=j=xu(mo%eB7fWx(WFfaaJ8Lj?^q_kizrZW3WRn<B$aW{=m^N7@D@IF}eveB5mR;P|S9Acm5bF8xG%^p)$z*)B}^~(npGvwpkM0#Cpy5Zs5?J$P0>b>+yp0E!NvVnH7wLlOK+o+Xy?w3F7iM0jY0qo0pN_rP0DAu{Kz3u9q@aN}kaUNO+u}aq}>wgnni6WW{M|p)qQZ;4?^yS#_qX8LOyH$tG?1RQY{TSredVP6lNn6Y|M~5(tFMn<|k`lK13;ChW@%nU1myZ=~oweZGLnaf`aOaa|X(%@@*-PD;g3?6Y)=}yiXSepbeVC-v2*8omD^-BVqdrRP@6G+B+Wp+?Yr@hF35)(D5ZajRbLA&EdQV=lZ|1d+<J-~X_-+*YW-$-er6X4HkATwi;V;{7*ALgz!re*iP%n2Muh%3yEcUQXY>WGi<8-d*+bbsPPfnyakei9LVjbeY<TRE~{Z^?H6z*TpCUW3ta8O1Kx#=|!j&Bd<e9dpePGYLp^CQ%(>7;S)raE;H$9*Kn2Z85a>pf0sFq1}UyKb*GF@j8aGJ94f>rfxt!{(m^0&*&~)0iu(^|`W0shGH&mK(P=5w4dCIy#k)g_-&(mx1j!TrSJfr47-w+ikbK9Ctc9@s?e3(g}byqix2uIaraT8~tkrH>{JcLrzgm29!IV4I!T@NL*Xyk;DR>IZ`8fhASP^({`aq{5q|cwer}C7LQT4j@Qrl+K}eKgyz{BHyIv3FXAFhW;l<prOTNbxP_lF2#^!JYu)N8?DTpgz%0y$)g~y?Bh)bSMZa;txx)#r*37>4SQE)uJdKdMg(iak<#&Gn7^>#cch}MFR{f^5gAhEeG1b*))p?@<=luC`{rtu5el=3-`?Q99df<w<-!J&{cDcG?Lz^{jpjv2tUzv4`ziIAK52*5iTpxZGelO%2Z0!;y@?q%(Oh=T2Y)V&9+b;di!&GZM%bz|ck12Ir&d=8--#lOE%jn<2KC6>3bp|79=VIpPxtM%5{Nfw9?+GtV7o%aP-rOIH9c&T&X(KYrDae<WXR~KzY?buY;PNiuR@hLVZ$%i*oC&L7?5o#y+TLVMu7MqWGRkTo6%doZ+}Fb#JhNxeWs9P*qiforb(l1KhEZv%pP|p(<^Rf1`^RE*ghk38KOL)eS?2mKr+|GmzCg|fjiRK%7hCm?<{zig6gCuqJiNLd*HHwH&V}Y{(5`LZPRDGqhIqfQ$@p=C#<TzZq&%Oa3IuvSRa~TlxWm?=O83_bDR^i$KJuT5p8Pl`1_Y2->DRbOD%PcDw{nn*c!OGuTlb<hF1*kY)NC;I)`y&osop;6%KCDBJKyEfYW(IoL#(FJO6ct0r>xK)X2Pv{oLtS2Bd^&g)d%-E_W0|wAkU<qaYSX0*J`ah2*IN~RtiLOClH7`2Nw9$og%Fn`$0O5Wqhqf!xs416$x5!5}`ttUTIgjb?5Wic|rgn%{Jp@Z@4n%+ih{Cq3^p!(SPzyr2(m2|5g~kdqKx^S2JtTSse?Ff-Eleis5cH-;8~U=Z{*#b5bYh!_Bu?PAENKU1N%q`%%4Igs!Eyvw0WBOX*7|X9w$*J(FnvV|^t+XmyVC;86Yst%y%v9Jn3s7H(f?(kI#pYRy<vC@=zGcd*nV^=h=Bmx}=1&SQ0!k`bY9dooT(=lVFCs92YO?gGM@indB7OZ;bRvHi4otlliA;x)_KN%Ipufsp)U`u*WI42sC%aURf{woMZec~Y+RMDdtnxXG(awW`6V&Ks5?wYa$)%AaCKx@8078#07r-2j7T`1rZ9eZbEQzB~N>!Y|KnBn3Xy)0Iz+&vG)l!BuL-ZML&GeIBMTwb`P-F@w_zcNXdT8xFbzfkCG|#3f$R61!G?wHgo3;A32Lz4NuvL8{NvaDt8F&!ab=5|4S|8Gqja(#Y!L>U3Jqe>4Dft{i$+pZKW<Pr3Sn)|Zrnkc}*PMT?qI2u#@&edyxpJB~i|UdxLcSOmJQ_Ig#7+92rkAXw;PAczgx2Rr+&$T8VMn$&9b^P-j;MF#p?w}k|)zGVCe=wp9)uHS(;18dmEEsRn8Km%#R!)L@7&W&p#iGc|)dkbP){=D7gX;REjr2|_cB4YoJ7xH%ag&H;mRyG9VXhrg33@)DRA&VgeoXN?lsnvg*uYf@8J(QuV(M!S>fG2-whf`u<cyXIgkNm#KJgV;&Z0>8<LQ1t`)I-EcZ?S6%=6SX4wkVCpF(li^)ptbt)*{+>N5wb+5IT-3V*LjWHRH1cDkRo?9%d}uVV4|ve(@*rhQg(JGPlO-aU!q|7-}u?v}ijGs9R_b!qJMI>BtS%mlIFzv0ea~L)*tI5s4M;5@yw*PeCmN9YGA0KNv;d!Hx=^wE25|yQo4pjGom+;t!8@PJ6g`Iz1lE=G5Dz_ChJPEq1%QAxh<u$*bE^G~e4o?4M@Ecbca7;=*OeOT~4$<3(VYbB9`dgw>#L{d}zl*DIqI3H5A#-_CjI(JQEgt#Fm^)Xvng1<nL{yHj3b#8sI`FzVi$ld28tZRL#TD*3s_6|r%UQ#Vk1a&3WcEfVfz8+vgo&lI_SZ}$Iu9weu7nYwP6jhKYqEUoF~x=Rm-WYLGxo9plQ@+1ac>yq{)(^f0#5VvADShl?zn0&GDXfQyI&~eM^KebEc-nTSRZIewGwVRE9V#E4W(HGV?r^&X_P-=y~1)tVd_PxJW-fuTcm%c$wPlK&{ik6geE)0Psn!jE%k;w+aM^2_+P7wIDaH`C++8SK`O?GrlZV?U|koA1AVU%o~M&dyGJYM13V(GYi-;KpWZAk0)sFu3?dheJbbaL#K^BUg;h}<n0+E^~fVy$If8zVA;s>HH~osfo28o~*J$1L$zjrfcO1lYk({0?Evh?bQ+ABrEf13#T`<kZK@%G^l1Xo+=F)6*R4%S>MCm{YAEWUL)s$MB!&3>XE^IWR%Ewuh*S8atJnk8cL7uLQ%e-2z<pD%1R+s0~oF#T3R88U)kS0dLgkYR<#)X_U_xl$cG<-QH#tZH5uNSr6`&@$L`Dsa`Mu^>|pGi63o<tH9#J_oHvX>xXlK-I=8>>qty{H@x$u?`M8bZ{u1fshuD(>(b<&Uj$uYKEYl^F}2u-fI<A8B<O5EevCk-OHB47{Lj5@2+AJo)*g6oHA(mffd)AUtq#nQJkRTsWwqqXba-y=x#-~L=XB=Gb7#RQt1Z$Zxx?aSHjQD`d7$2`RL|M1axflJRaR)VE&9pxx!&5|U+rmfZEgyU%Guiv`i1sxK<tWDD;Y;6Diq%N;yFzg*lR#uYw<3hpJ#(#+5c+pFN0o;HR0=(ZF*KA>J6HWYJfI$wXlIN!tv3iK(ab0wu1l_h#^G}g<??`FQ*+L&0q7`{AYUtgwfF2?ZR;PG+!Oy4T@TE$Kvx+jxSo%UKiPUrI~2`Ub+uR@d=qkIEj@)BtFkTb7@t=DYnM9BUU2^z$0$$6Fn`AJoo4=e4S`oqfznJpY68k@n`Y@&4}h|>hNyS2zjKW72NrA`;qRaw8Fznz{IR@5>MzjsmsPxV~JT@YU2|)_L1k=L-}NL$BvMnPHX2@RiR2qmUQ}a^`Li(#7DjNTRNWp#W^+@o6G)y?L0PUm3)PVO1laYjaB`Z7i+tnOYX95uNbnIT@y+>N$Yh_5%Vw)x0T9eIC(2asH!R2?8$8)&+oqCIveLwf3@RBOL9d(NLH0v%sTl(aa|dJz>E?;*V_fKKUss7_w*lm?AEEqmhL38)4=Z+&c)U0iR@X25b7ObI3(XeWNLf!>b1DZ_2XrLDUARQoe}pC<&P(WMRKJ6L<`|Qt8Wfs1DrY&zl$_~9ew;#)whNDV%uIZhgB&vKB3f4=yo(;lxkARI%R^bN76R*7Y)adN_?0lC3lW$Pi}io&G+<K-J3VE$~PCA+BG05uLz*w(c=57>(?2wl*WfY7>=8Z{;ubmURVuFZS}sfwvC0bHH$k{Iu8vuZek&HGvvTg+^&IN4<#aL)i`0&v?<Kp3#|KbJR6n9Ox*&U2)>*@vlAR4i;<&Ew*)tt8h^y(+Ing&c$(L|v@{k=-RucwZ=W4ZKh_Q{S$kk{MX|75#l)UsK}iP{ij$*b&VM^Yq&29GB;>6`_hnG6!?&Fi)RAOyJ>{oCZ9lt4tz|(QEC%#-^BH09O@r=Lf>~|;=`P0k&z+w+t{~Frx2~m=*B`0Fr!_cI=vF1>$J~xNP8?a<iPG!BQt)1w#$AItYvxaUe=+8olJ}?lXg>;<cVJ-zwSE0oSRrDIl)PbUz1%@fpLkaPCIz^8uPJKj!MI#@jYUv=t*Yx{UJW$-%XiT5D0|H*QCnr?vbEd}6YRb@>|26LxyS7c_=cedVLv|2jRL_)bq9Ap4xSOWsY1SgBEM^OJnqonlwXC=QkiQTcl!Bboyw*jkbIU3b62?bR>63vW>b(}LcOBi0-E_aSB1{A-(D2l&8@jhr_e&|Q~g#iCC;QYSV}}5*$vdHPeEb*%;@uDSB{P>Bn0CcERU80W%*YS==CEFyT6eV6ZL4#dmO3lkWmYtNI$*z52rd@cL$^EWYE1xx5J2|ySTw&`UaRnWAQdfFM`S<?~QVSWe;8JDolO9iRN^xwD*T|8Pf)ppMVp_5jVn`N4*bw_e?b^x^0-|XM;UA{LqZNCek)81*1t4>zeDz=V=-DlzTqB*3Q6~zpYQ?CIf*co<N#6^uhutY%3ar;{L48b$;O)3n)>Aj62RjV(C4$w6vo_6QzG!RZDaKP};qu{*g-VcG$g43gl$#ub5hibVfzoIgnjSEcmkixU^$%RiGlT3a3TH-)FPw@`=cO9?}HtPsL3=Lp8RCbp4w)4Y%c#6_;}5%;nOVT#{_QcTn{&VkATRQ$*tf(OI98p740m6nt{f!iNX5Wk8qlIdVNNgJMS%&dScL%H&BC%1Un`Tgv9Qn^FOmYWw~!^<Hp@^kC{$ecH!KzP8l6FH4r8b5mrSe-qZ&NmuLU?$qX$D&O4&L5{RK3wn0@KF22Hh9E{9RSAC(!IOyI52m_BS%c`-|F)oP$BqMw5yW4{D}UZ>&|@8L^@B*2{L5DFBUf(Oi`hQN4oY0<qUEzjPNqmY&%?<h+yu|7!S2UJ$c*2&UGs4(L`(%rGu$~%;jj%KPd6(0D<M<9Q@|Tyd&uk#WS#s5-e-9hK29O`B<?F^A`1vx+Z@{^LA#9;t34Wk8z&~DUC)1?=G9KqY=!BKMxfX0cHd1sqWJSlRu6-jOWqD$NJ03H1T(`5`l=~mSb~+uEQ9HqIc)TntM{|CFP33&cuvQk1v%4H+Eu*!9fDBg%IA)zRJ<?LmAwZ2qPtsaV!WgAoc>?MoAd!ndL=&UL8|xt5As;2E2kCE=llkIBCbg=F6ZWMO3d!_LJxDTPmR4W_Xx)3m+egYo;#`%v|jpiu)1rn<RZKTWVvZ0GPw+y^M`)u%HK|BE^wCQNHXn#jhxAyT(0jgz|}m(6j2>xq;%&G;ic6<HdF1PPX{UP+lveH@WogOOZS)h<aQ#5V83_@;5aI<pAVK9h|ImRaegwE>|j00wd!^dzFWJGGoO7kH3gJAMNCAHdIyaA*(Zr|;bNyHba%r2!<rBbuDZ~uy#77Szo~RUC+rF>BMq>J(}c1i$kKCmUfld{(nU`~TRjmdwJ8;g%^@-+rE8?)=a$m5O_#Rv_s>3{&eN;D#R9ADCT(;`Vd>X&nD2mV4cn~m8{mqfQW8C#Ny@zzmYx?@obAMyLVS7(w!mH;fwYq1sh?KD(I;mQ1!sJzqnXpcBIX@>$Yg8(*Qe=B<n#_t%5A#a*aD?h{3R5V4faCc6}F7FUz7a8bf5dJ3eHe@bxhF`w(o@4Nc}9Fmws`ekb3==*L4fC$@~RZ(vCnsq!qBs$bshon(3)Vlvm3~BqF1ThaGIp!4O1^wjxnL&W&jut-X?_q?})!Jz~6wc*&;FXhknaLIUNOzLvcE>*oje@EhW=@4|Pn8NF<CUop1rEfbKBqIg+hRODyRhOjS9*5j(U^m~_b;1fMILcSREQJ=Nq!|aXsSM6m~*iRCyN_)~M<|Z4mfeADF3$TX22YL*Xx4CM6oErW;cWPSbHqBeFdotc!zAW-@R%x9&uo1S)VWIrVVTDiOpqR}L-^ZF*ql&esr-IZIUr?!gop-U{kczjon^vnR8e+P$@WRrf9{ocdW#+agZoqrbK5>>f+zQwe70g1eA2y-n-j=~TKCSYag!D#MX3x8ZWOt^&(%#qx^HcXiR&I^no4;-t`mAY+Ap6~=(VN}H!mFcRx|JEUuv^YE8Ib+7Tr-(u$Rw51+e)!|X85APGw%cSm==ReeOK{n40a-qBuWf^#(`%b4E%#BpOi8uPFqn4n{WH;4XH1AoNO-~1NuCjZw?KfRG7W6buGt_hWV@8J}V>ZIq-Ka%!&kX`Jz@QIGD7}={w#NC5El%u^}@~)0sTpt?syD^rZoKtvkb>ZYV7*dQ_*GG>d&!d*6!&S$VH7PB~d6qL0|ETfjw%`v=6|$L{sKcRG2l9D9vH(Ox18-0nsDyU`6<I##OSDviw~d~`c4@3+WmL3Ku?+#!O_3@q|Me-t?l;>xVDN$HTZ-zS-=15r*UmyT4!wbu*Lkw~3ZV}+B8RBqV?XV_up-Z_H;<5HZ?!(!{)U5+=WYim+y*TVEl{i&SlqCDw~pI3L(#7^yd1a^q$!vu)C+_<$Ox!vq#6wu+JnTF!b)%c9<3hG{Z?sg{ydV%D=kYAaf)OH^htVpv`Z)jIPEj46TUM!h&Quws~>J;$4w*Tx3%}Mog0go-;-3FDeLFw8tYU^ccY>NtAyHWL4pFP=xF=|u5u!;R2+khJ}P&|n2&J=N3m9q0u=~PCSH-H_O9P9Rs;oNFsEjAY)rmx5I*B5iPD=UU6CwB&%HLJ&&!%BwUYhaJ=@6diQED4UUj11|37V$l?QrKSc02NctD;=}AzKKoz8eH=G?n%=}_2`bfat+=UD$oMo>yL-j9;pbk@P5M@V-~uVY;$rqF#LpEjk*!5iqB4tu!I+lqkZEFh5Lbtlu}>}Mp4&BJ1aaTq36A)d=wI=ze&?|PRZc2lTXJ_YkA#X2Mi%20ZT;)&GvNvZ;RCPI5*`2$gmHgh6RO>l0LgQmAb8Nq90%JIWD~50z57@g?%rB-nq?rsl(h2S~#<d)Tqlm5tbHbc>GbHgF&nNRAgN)ZPZIaCW8z!r;z+Nm!u2}b=Ee=bhjYcr2CSGt$nYBQv;+{IY#MGXz`ZtWC|fOIV%_BW*w<%(C&GMlG)v6gHLzA>b0ilM;28ivc0F?|J0tyyQdZ!S#gbjlZExC8<cQr(LWls7O%ivnnQQX`RjOXC}itl`|c^UX8|1q4fa^Ae(o{-1;yc}w?EP=i`3ng)$wPrmA>%S)HpFXnCrB=8c*+=BkX#08faGH)8cR8o->M>!0HiCL9my2TNQT4cOj{C5_`jV%AWh8<fJ?SFZFYu<ruuynT9)3C^TYlc4l;Ay(ZYbbcmuu?KJrvsW-|^NgKwPZQjqhJa01nbi$<Tr~SE^lkdU8cXsrC7mf)IwU?tri4(-w?w0myvyD6KsNNXp{a@RydH%pm5x)ZOU4TD;8?ZR829rU5V-%dO3{YA9GH00PvS8{-RA|puj?-CJZj3G5O1M-Ugxl4VEh(4MaNQfh%iK^1>Izql^tfW*Dv0hY{$DUSD4ekbSqtOD=Mtcu$GOcQ#?vaHko*uGdR7Yocn*D2*cVyY?Ow}Vr8SL<3bkT&MM5_>tGugA5iv#utR_$>x-Yt<w?dCQj@VfkxY&SyC%3WO!3AhP1JRXrs?1)^7<c6EmR#Ryw|v+6`@Oy_la+Q6Vf^M{k59@?twG<a=+axchp{1tw;6r8srcolBNw>RqjRZu&7sbY2TAX|O8=jBf_L~vHpQzDrl~NrM_oWti?>Os4%HT$Ps~AmqKIU3u+Ltnt_lP>W#{w5tAbO%a>=V0c#q2Bcz*RRm`l9wqy5VS-u0n8<A#IzEU#3?+qanTp8|iw8g&S?p!ZYTYa6k+z(4hkG}WCnC#|$`*r0{wjcy5zP4!G}J+_pagLAXG|63vSDI0M;_LA#ky6B|~91yt4ytM>%_wyWEjS&F3x5eE(?WVi3dJ4}2=zw&=`<qEe>a2r**k0$fyRYWiwX;o6?`HqEbmmdi(c<;>d@|6clSD=&5MeSLd+e4&%G>xM6u)PuYoU`>yA+`ph3_kcB|MoR5Gls%nV%I|CZ_iNj_OZ+!`_a`+p(zL7E#pjx|fxVLqDzSH{BD9d-R2m#9Mq32kJH3>bM<e;#4GE(~~LdSE*n+xLkc!)GtBx#V5wa$3<&i*9L+VV~AJ)0VT7PgqjxHV%(?(2GgmWz?xoI@1erGvde=RiP=93&YQ<!;d>7PPBXeP;+@{N>;jINyt+rNJy)5y?^LAHrhI7((nD)Q&xxV@is$yy?6R$pc@L?Ve@m?J+!fD1#$kS_j<Nh)inIu9_EdY`U$WutbdbI5@b@?2`z>dtV$Gn=yeVHBNoj;Q$Z+Tm*Zz1gyBzkUeF;aM>9fx5ER-fX@-V*9{%X}bTvsInZ0xs&P+Ti`d@O=Gv%d*imECR%o;0SKjmi}x|HcwTQYpH-`^H}9g{j2%pwmgI*ByK<5)-*3R!8kRDmo8E>sdG0ea(baShghv>9;v*w3O{^r_VbX;WW4Bq8mJ;bvPOAdzd!bz8B5u9<kn&zpBh@2;sfmoDziz^Av&OMNJaGssdY`yu1eRxjd)hCV+G=INtKXe78QAR#3?W&s^H%gYzENKh+_>U8B=g1-zI04arab-yx6?;;K5kT1WmksmXesXmFM$ezP&AG!dD?YR%z3y(w>|5F}?5+P<p(p$XU-AnAh|`J(E1HDf-qDB~qOB6c?l49Lj8ukN<1pedEJ1^TbC?JNH#XumZ3F^`eaWG-(GqdDCTpQ?Co>{{}<=}SFHTNNIXylcTEgq@~!%r$7oSbe|5A2{0#ubMv_cNVF5oM^ER!(>k?LFq`Nwymq_1`yd;#rx$DDMbXfm5w*Y%d&Q`tT(-mQlq5V>=WJhH6bpv2`R-biW~MnvG$|Uwpq5HmFjMN>0(hG*g5IE%l7yle$d;ld6R<IF7N~UaCP-jb6+#28cG#kor%X6wKjD)U#eS9?;i4UJLP}w3dNoco1N{%TE1iU``lcM0C&3%*O~?q6d-B&+$W+oU)*-ZOoEQSI(){bjOu1S7(uJy{`}?*Y_OQs+L^!d7wBml)`r2dkk%HrmAL+?l4|P$y2uaP_#@A`@qAXiZ;Mn4pGs7t(2vHu^Ou&iN6qNKH&4hq-SFdn_-i~l=5n%9$D(%!4l%Do%v@cewc*Vb=ib44Ll0+Pi00$I_63Lg=)Q@=MS$cr&<tj&6PlGeLR`?ZM_d@~3;^_bH2Ph7GwK~Qz1E*eO>dmx{RJCNka1W&mGpWsaZI{$zXrlVc*jHTeKBmyMSrs9VmMihhliv=5|}r)MsszCR>)unHj^aH+<Rf%IP4+-EMVMvs0_E0$#8>*J*I6LFgCsLPpsKjcl$#$$@W<OVa@lme#h@N_1ylNpyafQYel4^^z?pFVWq@sSBU$V?LO=B>u0v;fki`c3wLMGq?{RE%L>I4xLf*ssZt+<U0VExL~GU=f5HJpG5F*-ACF2O0KZ!<7`8_y$Lk<86Y*Z9vGzwRiB^g-jPcngL)kQL-~Gyh_yaM-qazpLI2hj-ogvfo>VBvaf<DhtWlD+9=W+Sc=8KdWeM<9cpP)y5OPDGyQojoGy(BhFraJ9g_VL5(mD$943t1g1A??Mz_*~2n;mptfWTE)zhJ)ttv_xmu<uG%e(28lU)f0dBeQ9cdz-@%q&<ggcF5=b+Bio@|vP+Wz@;H{7bEW+OPW|Wh-&(gg7NJdOF2vGuJJ}7!muIL=HusJ5XUo#{i+&F<Ocv~$U6k0f@fy>rWGqUjQqe7TJro+A#CHlut=Pw}<B{ZCqG4g6Tg!1EGx70i^y4U)Yrd5YD_wd=ZAAI5_do51z;0-b3A4X79AvMy;YTxX`C_(TstPr#K6t?D4<~A>ezPbiu%EV*X2q;mUh;O_yB|<+DBftXJlL(H^2q~+pX~Gh>d}O~01UTU&!Nk!S&)&{ZqpopJf%>&<|4RXx~T2fTuUExM3?GETyd0jX83Pw2HNJeY;@R<qI^~t>d)R$;G1UWa__eb>teN6B!%e#r>QD;2;S4|d0DNsiq}jk?bhPeq;5B6Ew(`i^urH_ClA=~)}gbf4cX6WO#1clJQ@F$w*$~iHGk?Mui=oO#`+ctJTQ=Jp?xBaO18~{D>g$kmdR?9PWb`_JB4lo&vnZs>qLg9>7W9t2JBWD@N~PqhxguZ-<CBaI>qK#>^!z>q;8M033ONXSR>N*V7R<qYP&<YOCKOy0z2wL4e>U%`RrADH++;Q{m*_z&kN4o8)~bH+7(NFd+bfKBqd6YvW*C{_(R>Uctj(Vmwo}XDqTK(uhD$>lYRTRy+54o)%&lz_v}&}SsDhv&#x#Nvk02|AdG+k3GY1;cFrI#A&^4`kTAb}C0tRar|-S*v%BYPo#`nmGcqzVGBR9-#CDp7+e*DSw8oH_Uhd)JF}RWIRuDwBIs*<uRGMbAQJ&_l%eq&Y4Py%)j<1Vn_0&OQ)drfwriZ3sVKXgrL>~{`ZSk#06j68|SvBQw&JGS!-7LqK{q&3`8*<HPox@^ap{->+8<NLU`P9qGqf!wDpHI6?cTk|6H(P1k7&g+&N-z=Zixft3Q%Uw;>O8p}BDP&S=@1ReWN6F`Q*%E9%Gz#DXA}>dMGlD;Y4>82UML>kL+yhnXeBG*rEl2T%=dj7&pwXSxTZF(u6itb_KVmSC}zf?ZhL7}he4x7_Br(F6Joi`1fOwX+E=Ogp(??vYNf=<sr9k6mNqGWcM0VZcpMt6>g}QjT(ZV+j*(lkuj=LIRv2vZ!*On~YnN*)IjaXZ<0+y9dwFw9l#BKxRH<Bm$5YaB@Jvhqa2Dxz99@`~<PaMg*vtuSnCgmKD0&E2beWB+ZNNZkW+)f4e2!_FQhKG;A9adZq8c2;ky-ZnttDPhbDXuf{z4S>MF8f<Ld?UZCedWpAd-bzg%&Sp6m4^!aeXDVN^tt(H>_^jnaf;yqGGF>hk6Hy>l}I(bX<)F<J~bINumQGx2!%5jfYi<Q@QR$`(31i5AvfmG2EwGY@td3RJ3#qXE)rw9L+a?5(!Z7WO*E$MlRmtYW}EFJ!XtLcicFOtN07%#D6U9QawGhU2#(!pP}2}_E;s5M-GjdG{lY*eIx8p4i2hFi#%GF4jVWR;{`!qjd~#y+XOLkP-$yxUsP$)p@3S8aI1LOvsle=-vV;NW@)=6U_X7XKbJ6;N3}Y(t1tX^?&6slZEUy0ifP+h%*fqR8#;|jUE{P+s}2V<;AIw%?khz(GizpY{HDsp=w2lnY*N)MQS#$??*cwtAjcz_G&FIZE6n8;|47Zp2HPbM3+sk0r)M&`4i@&wR0^iSl4Pen<EP@JRl2)a&q{d4-X-V0Y!mF3Jc)vb%K88{HZ57k5!(5Igj!5f<KzyX)QT4%ooi1AE3RwAlfiblI3O~|d1`NKHFFtNLlK{KqJgI3<=$VFR7-CwwM;+PFXeX@3$~(EzY^9`%8FbcC)J#m_5_*O##A#>DpoG^f_zEEXF}yNZu%AfsgutLDImrQB|(DqQzF+z{o7cFReDC3fU1WUU)Vj0W<-8c1>vonM8SA{emX=mL8#+jHv@cSg6Xm7apt(Vd{Q|?QjQ1IJA@md(54tmBGKMu#zqb^pC{J}6l#(a16EX>&K1i2=erDoawQH;^Ux8M`kVGK;m7TGdxqK`{9sOx@)>P!_b6d1U9<MAkqB;KVmv9%7$D>+b5waDmcpFYX_C=hVs4#5Nl6qo$b8}-$P4ijor$9io&liZBB{!ii~s{=JzA>U<3l=dhM07#SCJQRCzD!udKzx`2qRDDo+<u=v}y(_bJXglI^qBw`jeh!b0b-DmO751p4KML^GQcfk8!2HM;f6uIb5NKRNDffa^j!_vlU@W!^Z)AHpywdX+vex(;G3ROtyB6VVD<rByV@hs#enwFGvQxhpa*MLi#DYs3;TGTfh>vE#Pb0QOvXkYOoqTdn#5p0LJx;!4^_2`?!VdcE1IC%9TyCm%**39<SqhKAWgY1A6}~CIq+a(2<5%ZJg`|$9l22q@%k`(FEnmO0`OCCYKzwyXeyl<z|YHiggypO=4YO`d!gp3vshhnG6&qSvM{O7{V*{(%k5b)1iY8SzytcmueVtgAy57#=HSDzY@tM%6g4+aJ$@POO=`K#C#on6K<_GA+Y~YqP1hTP3eJvGQ-w~Tr3UkG4L=1F2!kqnwGW-d+6p$DJvB9cUMMr7^~>Wz|g0yxo;W`P6M-3GPx+R>#2`JaIQSFARNR!V}N6_C~RfQQ*qOMWP@VOtr?D}$3$>Bnx^D7nh2|b0bB>oe)B-+{a}LiaoJ0j7}zBxbE#+>RFV-f*{(*q<?SpmP35s1qjRIECyWJ7(-H~q_4PRo_OZaCz4w>M=li$M&lB6${mp)3GP2p(<`T1G3&6V|6f{(;t@>zJjFr|UsG|D(0q^i!MhXF6G$dkJF+`8ls0l8a`i_Y8`+;bSLG93?3-)L*%kJl=(kh9=TUZ|Q&-0Pq8@}bC9kt?0H+dQ_@K`pkXT7y9K8%|_R868;ZU*{6c+=`vyuQ>15iZSu(5M+XB5^RH$t`NeB#&IcE#0!TQlo*_$C1fqV0#K{!&tX=554Z8rVA_TSv<c7)C+Ad(G^e#t_(PKm|^7#g|semt`83~`pMf$G#S;{J6;sMz*7Txk^-|Obup^9Dk+&mg}G0xn9F4%swa5_R2rMvU~$Qd7ps7HS{1uD49mes(OMPkHJIt8Qspv%>bBLG2z*iRRC5#Ee@bL48X#liB$1<y1MbBV{lSv#qnzad_>v5SHVw1g&JC*!x3kFhF$9V0Nl++^qG>+@Ve*dmd-f^)@mf9TjBH@6lC4O-EJpVVygn(<<WhCGbfOw^-A$YfT$3|Pjolxsp32hGj?Kq<b~m|nu)%SqY~+HSE*$lZA|UV~yUvC#xHdqJVOy#QNG#D+XOC-(`Z9QFq(X<064Ph1%c|(a%p_VbQoypKti$=0ya%@d*`!d<4BC!OWU@?Qr7^Q-ak(De*o$o;xAx|ilf+`ajL)*$c&RENzOWM{Cvlrp8<fUBBLDIZQ0l!Q&`G1?5L*c(&fB;VCFN?~b5=JaVTzXw$+~#C9FpAhu~dtUQ;nKfEzOxA8{Ll@0_m4LSVQ!s@~{LD3Z2g!%-x`hsZRxuh^-+MEz+q=5zTHyeR?RKy29kL74gP)umk5iO|<0mQkruj?zK!jhfRicWm=reLvnBiGP@DgCSvU{9-8*RaF!H@rf1uUdh_|!K~ZX&aeQdddXbi_Dds;!8|pIJ4^;FKAePU-VgQ{-TcTja+oS0W@)R;T2ridb{^LfV8rU`)-cgHfqr@)9`E>1Bp+ke>X|u_M{IOH&l!Y}A(h~9fem*ZpW6U70AEo&vUmg%ha2RSqt@Cs_O2g)oM|FglOtzDFhpt9j(dnseU!;;4d%h5<r1=v3sFL*#(uMJy9^f-b@>qUsKFf=Tm<}I(mVYfC*=9^mMPM+J0f0t&xQ(*Wa(YZOmgQ%idQxoGYWeu8=j-BZ>R61g&FD~W+Oe%V-~a{8@TGh$6dDOCoKNd5&pM=I>v=SFNw(#7WZ3Nm0;hVV;O({>B$iJL&noE@Ih^rApHdSkp}zwFdzg_BN65iA(~-A8iWrzaPYgL(QX+z%FFdEFg?h!uhuR0kKOX}k^L&7s$P{a*QoEef7`&gM?1e9y6~*AXU?ssuJK%?nLjm8%4&IfQ4kA6oyQl~W1tvwC;L3Di7Sk|pUPl~bb2N6z`h@_zN8bS<g2U$<@B7hOqb`W3Ef|-UzSgw~)M*6qU2C2_X7h?IM6_CIimT~x_s}oJ*Z?{8dFPoUb`Z*~LuTQFhDys1ox*?z?HCPDT4WZvY0{!sR0fCpT$@@x-cppY!Np>n5Cbs}j5TZR6I$yz3Yp17oCxBA$-YDS8Gr}`%3l`!O$Ei54YivJpKbfVEU7utiYfku3DFs$MV!l2gSJ<6y%rj#n>$JcTKPb*m;!=Y?m+E=Lj_+d&UtS)LRDT(Ws#s|X1bi#^v(3gTQ*ipASDB<3|b7M!F^o9U}D}A=r*x9QEJ{Qbd&LOzs1EuWZdu^x_B>Ggqg#h?T^HZqL{@qC38n0mhZ%r^oq*rqqYB%tS!h&!%^G7A+CsvNBottVfEmrkqA2A)p?q?luXOtM$^c}$1m5MUm%#ch{sGQm`8~ay-w#@1xvQ_bOQ7YxM1;|*cstjg(#kAqDTzV&sQQL6_AhlH^mf(UWl|_J$uTnf$XEC0<sJBqr)sw^X;wBRBON$JQP*9L&U}oT+e1qcrS(981a1ZK0ax6wQPH3XZLZT*E6#?DjrbYnwyf9swmK4AdW)I2DH}(%tZjV!-0KLQ;0nZO+%7<OX-c5{lRUsS=|7~qkrK9>j_{U>|F$CJlmJ$baMOPJQVV^W(hRhTvumaL6wpSKgk~P{ZY#1QE06U<Z!sqw}QBmngA=lx<jU#zZ5Nv73zoxjWuz$plPOA?;W`E3GY`|YMv^sts|LIwpDmP&iHbVH<oYDR)2~?*$YHl;&=zllLcxC;rm{pzVj-Ju|E|LgDG~?^Hy{mQ6)W7j+Tvv;9G6Tt*7TJGGTwDZ-|lYK3zycWKl${)YSJ3;t#c6WxOJ4niiHqRmC3$rBQOGDTTgol^t@!bR>@_(#HXiNfjkCa>Ui)rM;(0`{6DW;C)RxQ>PC4%I6<*^!a&}VTS5np$^9deG&|{fyH5ekjB&L&eF1-xNI*IQ`QHP{dRN7ZQ#5w7}APM0-wwY;wTEmwqS9?U236uJvBHtC|>r@93}`CAb;31x}4>*n&%eL!L)rDd!<Xw4!4jFu^3Nwpgz@R0Cp(!R#JI{m;jtfO^{-0+JTSFlLwlVu@kW`puQF{;HrpH0`zpg+=xoCS^0AGJzn7B%Zx<<s7EMLui2}N7E+ceWzVILWgXoYL&Z42&e?QpHVWqwRDCd~<K=nyQ0~XZEHP@N30-#lyb2Hto1GGUkak!@NnqYnx=gGy&wPbfU|D}|<BPPg6zXRnGkc5+L)JXO0G0Ezz91RjEd;nZjKq}&5b%{7<~V_SXs&GSC<&BSjd+)m+RQpdHj=E))N2jgPX$h!UcKWvHa3~(3&c-beIpg*ZO2x_Y<&gp=!8@~dHGy@qsQ|DPE5kl_<S%Z=AdFPRuT(^=%8n92X)v_$UuP^*Ugr43=H=+cFMFj1*5Bvn7|3DKJVD38zszB^TTMioGpNevyV*pyo{KI40nMi%5j}zWiovX&)G<$@%$35^O2t$Gt4H^)0)0qtwO<Uj%(SvUBsyPww^n+AUlzPmCUlkjSG+%-zNO?WDZABAXD9nm+m+}EOm_Wv54<DozxrIc#BQQWpmE=0&s4<=eCdMWx{|T4s4bzqE-?i%{gvJNr?_ss^?408tm4L8?!LsOT~M;s4P^etQyzJGK&rtWvjt?cP%Z3FF_fj<V9n&3!~d8e#o!v=cP@+@rd__A;CjssuEW6i*~5fSfG0e2#b)v8pT@1;n+b+9#4y&NBh;7n%$&ksS7+VoLLMHBx)J3JYP*vrXC%K&B`fkZY|*0Izf%qR^V5Yr;YJ;@JM!_-V?()Yd6S>l40YF(#lG^t$+%h<2ZZ>3Lvd>6HcvJM=Lww?evzZ1LI4wETR2%s|O)u@&YYJ3{`yEf1;OIn$>c%A#@9Y5tfyz6viwFI@z)MLkaEFfS}J-n~i3Wm~NlZ;z_AZ_metXH|&|6Hs(pvqatv2x%TPf(xw|I37cHcQ`k?}hBzG+#aNx&SEKuKp+e|%Yu|AWa4ZT|2_zR($IvPk<vR#~GRIo64|Nwjelcs0Z}sX#1+P)Se7aVE)48HvYl(AX*y*F;LDg55%#m=`iD@E=;p%MDYr)>d$R8Y%VgIUx=Qf)LzlBPLW=$&0ee}6v)1sElD`*_sjT$m(T*|g@^mu@hDQwPAHh-v(m5|S04n&6e;XF-MeUou3<DpaA0~-$Z13VZH4NALGr|W0dQ9l^d5>DWr^QD-60`Lrr({jq6EC-XM;kSjt>|8U29M&Y7cw}0`pB{Nd=Hu<Qx0Lf39dVMJgT0<ITrMN+1g8;{2MWZ}!yr=R+ekc%u0p4xhfOgBEQ&@9!f|?xV0)>pT6Mo~#gRkL$r2LBaN7HHJqukeI*FHk+R#|{!!d852ggFCdO64yTFV28<`^hsD#UC~WxSakvgvry%v^2PpeQ{b$BLdQ5#!OwR6CXug=lZqS=hcE?<<M3t?nb^XD`>3AN9;>QE$OZ@2IIARh^I3)dn(4Y;i}`G$r9O7>(En8)e}}no)xjj8In<+mZ`iW>aB?Sstt+-fmGzb=YEL(<5u_P|Q+}0|5>CjOGUy$yp`6ethMUKf02knqR+o6PNAoY_oJs?q~DjV$&xXKIu`%4VdT^IbSFXWqpk{FIMMhvIJ8t&qD97Mq+i@^8n%Fj6~NL+Rx6J92GE5V;?Ki_4u4BEs4i_oR5+TD@&LphI>Lw9&8rM4eGXza6IqX%_NW7;1CuawzE-=+0~ch*|@Ww@eEg5pqH||7Y^vYf?P_}${+U9p7OKgBU;lCv6gLTXdrdY&Rfo2GE#|I#*@}AU);2NeliCFV|mP36WAK*fm4XsnH3(1Wiy;rZ`7oXk7D~<FmT!Gl8P{c=n1?GcPGL(1rAv<Q#(D%ZaUhud^y$AoTrzWjQkWkqhq7px*rp0U#uk6O5sJ<+}8SWFH+NsrYr@k$JS^MX;U(3*jq;clS6)%Masxh);ov#deLu`0ks7_FEiK>g%y3dX=cI}`+mG@%mFWj`_*+KyobR9vS7>+2e$Lk9o=Ix^3j_G$C5XvS?fNOM7e{e>?4Qza2QhIaX2+k^^L?f>M@J;F4%}o`V9f~Jj&^S_Bb=BTo_`Q#<6}I=uOJm+#XY~!e|#CB(qziQVfvm!NgWVo`f&B05X$Gvzbmr7um8ppL1l0WKB;<)rXzY05D2%j;q_P)oAXI?oYfFpFZ9#oG+`Jkf*OHyq))irzg+AE^r&Pr%=J@VPze5P^FTWdcB>!$drpA89U9jKnJfi^oiM!H8hM=wq7tXU4f-j=2U4=wP-rEkL+XdVeeTz<lk;~!>v8@U8wx=)XJUG6%Oz6M1U*LGDSGPO?#Q)kks~F)u^r5Xs3)+3MQn$E3h2wq4V(r(F|<Y54|-Ut^#$ag6nI;XY^w*XKm5ZlxECst-SK2{XKaa*vtkDW|wovvXm_ZpP8JCUkhf!t8zMpjx)*nx$K{A1ktj^x?LGgJBj#VI-E*{VA|8|O-oD2K+&9BEb{T?xgX1}p6`-gZ0FU&#Lu%n=frW=+4Kv{I96xnKr*(-cjzg)Y)gHP5jxA%tXAlY3;}s!g=DxjQ&?S&B#T2c69xN8U^5QKWKX{ldp>r4-1VyhD<bw(&J`Buq$-2ibvPR*s&fVLmL03vSzJnW_N0xJ-C16By;U@w8!^I&DqiyIK_!~4RJ~GGdRp#$bi6pt3$dXQZ{$XL$s;0QE0sk<17)7o#PYi4=<7zV*Gp!b3=L`N3{h`ia5%vOh?GwmrJAf}TYa9w>rBs+5z3|>wSxdM@2&QeVM!E=%H!Tr*b=^Y>6~g$l=&*>wZldaFSOxfk2VK6`!vb#D??%K(|xf6ZR7P~;<BmRI~lL}GmCfy%8G}jvK#FEa=s1D$<_)v0rNmUc>+=I>Uo1nC(LckbX+HeN_O)ZJS{h!5Sb+mvFx2{iTO&Utz;=l`owK~ZPFg{oUDLo8ZQ__c5=xM1LMB1l7&{l3JZ`POHF)HG&qVYQ$%IRO<*>M)yG?PA@7iGHfDxd6LUOH)tR<uJUAhNgT>@Ro%x|yUd_{4bG4lW0>y2%6F$d@l#rz%%Bwc0MQdLr>(zzXYI-m!rQ#kP=@Q3qH_)*6m)hQ&ndRVz3Y!E(b^y-v_}bdH^i4WdTX}NDAdo|WnsxFH(%MPK!q5Wn`criAxD6A{iIM6ew0~ff(s?1!Z`sN!f3zBtd?}o}Y#?tHI9_N2L}R6NAA&cu^mwv*IkXbGcP&yhRgP5S-izT&JkKkOCcvycTl0a;@Oars$MXeX)nx%x7V08qV&?%O!!+na6k0v+%wG8@Fs)-eh_njF1r0_KJV@lMM3nA<j@D>dfuhu0Cqc_`I=R)LuK~^e<K?=Rr2{QIHtVZKBn$Mg@jMlD0!`H>eNJQ*<HQbY;kjO?UH6G~6<WzE8Q>%MWJozCAs6s3_(-7(9!elJko)jRjv(k_uon@m7sj7zqj(&OT3WGl1wNOzHyBc^Zg&S|c9sv`U9gQuyv_QwbupF5rJU+|Ca0v7#{H+YM-z&?WX@r35$e_o1USx)X*<Fxhi&zY!XoG=)EURLD#DU_yfUJ0dRqO^sBzdVqaj<ZHCnPSwbRP8Hg(o25%vOyoP!%5rN}wYjzg-{Dvz-~p>VwTAQue`IDBfi16h{V)wqgD#CQP;g?8(}tCNGAX>P5PPq2wdFS0LR?DH_3lf!F8RHKbep@465U2|Go?x}fgso4HzPR>^Kbbr!pMJ!%B>SzjkSQ=&}jQYVvK6uXb)?n2LA@iY^teJ(TRz@#yrL;{Wb#+p-(?hBh-=UK>!Ehuv0E6A`eA5-;#p2jX<+q+tmX7f9Y|)nX{=K*LOk$B;u++-SB6A5=JJHOXh#V)yb0&prQhpvB6k|v@kP{9G#3IH^|16oS4e@z)?yFvyv|WH^U7I?`alik}geRMvc%B=DLegG%Yvl9symXjhSE=rV2A^NVOO2AgXfs1>a{J23e~1-2>QodnXRo7J(3uCL!sYG`%<l4NeVxPoS_;uURiGI@I{Y|eWpLHp)w3O6LkfmD4^iY*8}x&hMXL|%7A-O>8|ys31}9L(_(q@|iv~O69%IF8$4Li+_Ee4T72<xtqaD)A7?<-GYRS!hGT-XP(nl19{$g&;7Rl5;09Y`KB<P9>!6jCzwi>mKx4rgt7n@db*>1OckC)8Z7>`#?LdM5~*@1!Zz))F9!&Ac#3f}%{hc~8%63R~&$}R$S^Lgrog*LTe6(DMW+^8q!bz*o7@oE7;2;MwSrgFRO%hssjV*9*#?gV^?-J}v{>Sk|i<;nHp)HqbF89lR(%LR)|=wUAP7q+TJR-S3ee(u51IF$36hutG%1)r{E&`v>Qe~VQ&An)_S0u{(h_G}Z1N8v&NB?^;nzP2e%<4)KasRa)UVI?K%t?;Gud>%5N-{|AoHYm1-tk>S3RYcPRNcgxaTg<*F13|yltvw%X$e@PaY2;}Hz&7*sCC@SbYHyXq&W9Nn-V%MhP#9sxy4g7G2?JZG8U?q!Y`U^C(yCX>(=pR*$2_7hRE~yM-bNKww0r{tLczK;^_8SPus%JKW2H(gT-%!dLn#^}qJ64LmG{%UYBL^Ky{J?nBh_iF@7YFjq_K<SydzUem-r@KROOJF%nf%#Uz@MlGet}v)gf%Q&IWHqs0Poki&9r8&T#Q@J6}#5IbgV4oCDRfNQ->}>dH-aKCOi>a52^Q(fx!Pp6bc8=UJZ|;-T=Ymk%Q-8(V`tNUdzLhY+_<*P_5m*NUBNPiV!v)5?NyqPY2SwyBJ`%3(BEH-v<j*`b`8EJHjrvS$1JaaqV=u$B)uX(DIiRIE`jmDcK!+w_;#GrZ{p+5KRi0{5PS#kavaJ2!=sS@OobQKmW~@aL-^lyO}%#%#|j@Ah&Q$(7HE+-%+%RJJ>!mx4mIRuOBp8mAFvb~2%uZTJ{Gj&>ZY*;aFKJDzHFEA`3ybI7dFxah6P#6RqKYhY=SQ34yxNJNvxTzlRg?8A+IaU}}TOdp3$PXTJ?5Ty|*qreO-sEMcOG^n`M^XK~WJq{%SRFL%`4UuTRxvGW_v3f3K>^8wX(Lm+pB#{88t-;=sJ;<QTmwLk`Cjo^-n-U8nH0!qwUI32qS^kVgDQsF;6JdTZ12>O4;IYl%LsDF#0vVu2p0qwDiP>OA46EM2nm+8uv=}YRWJ`;OLP_6B2|RO8IlDkmdQUmnb$B-9lWZ&BReRJb(K2Tx;8fn>dnn*|rJ(^fsE%bwNpMZEn2TjL$>;Y($);Ci>~d=xox=vhWQQVMZI)XmRaZI#%NyKReEIfdI<uy*f-(h;?fF_nFkQ=zi{a5;32C#0KTM?aTx+!tS4p(WcZZYKc2vCt+G$VdoSAB|M+RrfRt><OXAhUP-bL+9Uht&zf&E&Kz@CMP4BOV}7#Yoq8srNdJVs^N>K%wqj<4mwnilZOUIHu#cvKSLP%zmu$x~*sPMn^P2jvs(ES<tqOyaQgOwdZGu<@h%g70>gkXY?li8C^h?J#;yAAN(zAX~4l<U?G?g4?A%-x=H9mNF~-6?o#A;CwQSF)~3Lfl-byFF4ihRbpzJWywQv6ev~M!~z{_ds`??#DEa<`d2=FxdC#+_NX_gdA9XQZ_pn+s?Qk9!O<PeLV8rAip|-v(%y7|sxOgf`HzRPCnRgl0Sp}_8`&GP+(Np{p4Gn;qP3uak}g=AfQ*%UIdeFjFAfk=ERx-ZI<NRv-czzl`chHRF9a*<W*PE|$>{FnW#h1(=9jCKR9ixez-5FZ1p)KyVc^v0vy!cfiGC}+Fdt;@`q9{Qp^Ou!7P-XIoj6)i<~iI^gZ#2)2jM(HNTm%O8zaXw)7g&Fd)-59ktE9G2F>+;6N@h-L=0I{6QX%~hylVZv^7L^*7hYJE|mk+$s@(kqFEpvQ&|t%_x07nDcM&7Bm)i_U39++&l@|AMoqy(MC08`;#pphYfGp7rj#3c*A%ns?itaDTB6p9bz4hVEcb|VPFfZfG27W-^v<)l)R7izg`ZpA9(jdLR3&q5pp>0+!GQbfL$flgLu0|i6Wl}!+E8NJCXuEm^?dfMD`4Y_sX1CHn;r60XEB;CkQM;bp;*eJCnO)~rJYFrI2(u6#VM+`Cz<hRu^SF{{Jyy#hYma}*V;H}DKI`JLl?q23iE!d*A-U>oZ3W3`NvnoZc{LMoK#A!Vu77LZ<kGMnOpS|9r%(&4$E2?-YwgP6T&1m^Lj5DOf~COJnL<mU`;KiPsMt=>N}O<^8_L(148j6RoyxcL)cDOshMK8b6FT6u--2Z6hPT&+2}MyBJKNBqc#s!;^p>+s!<r(K<lx7dX%gVb3vczT{KbM1N%&+0+*_!7Vt3pAT@-s`f#@`EvU;9HHTJm9|>Cv<{4nG%o8MI1`ygJ4H@z9^8!RpX97QVf_n?eEemSM7)49-a{e+J=KQq~*bPz=ujJtUu9My8GRt*FTjv`2zyd4}tzwI<iMU$Q)5T6TWSo5gz4{2kH_1*{$rbB(57>n^d9kKUV{<%V*jmd5wPq#b>2Ya-J;g?m(9!1~cc<kbn9c^hhdQ@uGxNn|PvVgspBzF!rjRU*$$8a`VvMp45OqE$uLH(?D&sQED?xFg89l7D5IYj0^hRxumg1of)Q|ZFBR%gYHdrP!8YHqtC_Rs7_n5JxD*HwJ`6aZS%o3Tz8lB0sp|!%%;4!m?W2)7oJRrfLi{oH1heKQPsHvmvSe5jBc{Gz!2MGzQDQ+)L4m7D!`;dpf)CAJbdD4Eu_DMCUxMsR-RPo7O_)$hu+lF^4C7muzsseYVBv~ytj)PhuuJ)r&?33)Uy}B`2b&|a*dy;8Vn6;KNk2X&!dAl7M`QZ#Fmf6Ipsrm+B+M>6dEe)gbzzIo}iRv)6tqKn%5*30HqkBp%9}Y<)x!Tu>*r+CIiFj@q2({<umOq-WG1goU?ox$Oa0bi-AAXPm$tqKT{f+ia6YWLaLspR^KN>DQN-0$K$D~fGy0Xi|L;ak1Jm{W7Vnv2Q(^=5=i*syUZ)TbB@DLO-fvK2Gjn*V*cPy?^<e20$`H>qZnBlyfK6_&R<D8C;uzEH+X$_HJp$CRTO);@Sy`jwrr4MS~2K9oK0Xn%@`@E7S{rY-R_YvckHJj<v<1Ce)CZXt==7tD(Ipy<$wnsps@nD4w^lGPqxzFyKxuTd==t?qI4Ka&!+~;ZUXXETvS&8$)5XsOnwf=k}p#iu0M%DJN%Fv129NS4^!(Re;7VPX67%es|-s|{2nLd(iQ6;Ith5PyeCzAL|z<}1u(WTF)>ZQ1!EBLZekP7w|i&UZ~`GOg~r;q58LSEu&F)_kHBH65+EzEJX8+0+eZKwI(Gs}4py|dV=y)EGv8o(N><IFmc`$@rs4^tY`x9xc~YEz|s=CWhf<E%_4k$t420(qPaFB$>C2tJd3OR>~Nv?+x5paY6&+Zxdxp&UpoeLcX}i!q~ctW}^aF9gDfztykPzCw%R7uZadyDi_i7)(zzs}hPmpE>X7I=bHt(gx2_CAmqLkChN)9M~;0p;)zcf(s3_7;bcTr;#z4U|PLbqIQg`Uk18tRojDNGuZSlf<t45PsZs8EgjlQfjkQD-=hip>d9^HwDb;DMQZFTl-@}eXPQ<-wxleNwQ{#`Qbz9Uik9gCh?pST766LLo5}VFJ6BU`hTL*xc(M%`y<^t^F8lT#ui8|IjEcuiiL5_r_1f`rcO66|+p{Z!$&>_c!*OVOMkGEs5r<OaL_U)-0lOJ<+?~FJ6!%LNDwI+*zAtU(7kEeb5QUrOFSAh@R*wx}KuRmmk!AqWLKEmA+|)-)D_v=s+sh!I7BQw@#m=jID|uN+i*$a28#SdcGXoE=P0=fzX#(P7Whj}-o6*WD-Ytg0$8j{AsaeP(c1R8Qc^iqvFyB5lkZSE*C#dDEWoxu6b(1DbhZ1wvJMjY9azIFh+Q%hBJ`4DSK-r^XlQP)JnV_#)C;C1vK~`de$}HWnf&r>hM*9rIj;Kx#$c3RGh3I0u-6_MNv4)K{YCXs77_c2BNWG1jv$}Ce9}}~rC8mI$!rO%$9eP~ZO$)x|P+c66+&-BPw|9ehf1-KXq~>jtJ?vS`l85+S16xmMc(E#ptB%?S4IsTQ>)9?u7;${!W%XLuGaW$~J@!DQz9CL5#W0}iE<=D91zaAc#ZecVh#@*T1e?cP*xyd8a(o1*JuUyFbHE;#U_QB57gnKZqoU<(5~Gi;v{FwV(je=tlq<uOn6}7jXRm^*<EliDPAjC-I&*olo7qjOr5QnHDRoV;jm>UW_H>u1-kPtQd=na%aj1JK<#$~3@%m>Uzgo@6mdNlKrHmVuZqd*ZM}|ut_0(MM4fv{>N!l~{sCP3~9knj_*ib7ltIpK!pndW6dxeFP*evo`oa4+Aiqifi)L<GF4mvEi72<eFJ<@kUQT3nuR4ZRx)9!cgED3Kg6AnVWhtQflo-(i&=Bx3x77Za&zkd#hy1w-KNIpyChSSzz?ro|oZ+3^)XhB!_?h4s8owL*c0cywwa#*MjJ!+H!(L%d3M)|trJ4A!)Qf{`apld4~JyZtAz=X^+hg@Ygv5<48>fgqL^^y@0G<(>MwGPciIktZ0r1$~@lG(~3$1SpfP@#WnA6Fv}z?z$8R}WQ<-G0Wa&y>hXFxRl&Jkn$n0%fW^nMsG3J{fDLWNk`7p1s$*mA5NcPmXtxmm85CZ%bzt#ADbCQzm9~6)7FsW0vq{C;UY#<bwJjZA%N|1QyyzCJ9Z-k1t@rQ~>6r<<hSey2WkXpY8b9o>oX(zzA(*IB&Av-jCaC81ol=QxQHPy7zQyBKT&!S@|x;t{P6sgX|U^kTHy}%=>_u^aLj1Q@1|kf<i1>YStdU6Ob*<qIG(I&@YsnK-VfOBr4H!30T`yn%C@@Cv3-fS__-=<J8EVx7BED$rksiSz$=#vI>DS*;TVqhsuFU!JJP-#TV43bM`VggxZ(<V!U5GT#;#RXK!YvnwlOeZML^eW%cT;tB}dr;`vh6N=k%E%u+rC5^Y6;kl9EK!RV5p<1lX@ab~v36ar_=l>BMb@*jHM@@hRYkP#~nQ{%pnnD;d+-J$|QI|aj?lV4UF`K_4bJ8-knJE@())_l0!ZSOQtE2MYZUB{^6pq?E`+h;B@IX+gK=wZ&K8mCnEvL2Cnay%BqSrTdB8x@W%adTjX*;;pCZcjD5aby$PjK<rrA(d-Ty0?ecG112P9E>&2+5)tzr01#1(MnB8>F5NSkMVSMA0ObV;uq{1%}-mWMQ~OMCWcg_&YvdnMJb&JnX1)E*UIN!Z9oo=s>8n&HD!><<;NeuA9%Ol@8MW=)NS^P9USeCN9F1p{Dw0utuSoxQ~#hi@AQk6pKqVXHHxt$^aWiHbx<9D;J8dmERO&9^X&(&D}sniiX!6%t4V@Pia&lj%{V8BcaR(_QLLlFnTQJ(d4M55&VR56mJZU2q&mRCZ)yMe596Ia7r_tU3;jLx8`}T+fp191w<l>sP-Okb-_Bx0Qlc30O`-HZT^yAM;HW>wsz3hz!}-DS9joci2po3?{3ttYJVa1u)saaIiePtYYAnk%6iu)d`G+$G?KeBP(~o1tQN4;e15>Y!{caKK<Ktcvb2%DMzVvnaD2Aihs5mWEN2B=@C=q)Bx{`;Fs%UjIakn=ajlT4vSP^quk5>VU{`q!1|Ga(#Zy(Rjsu-4gyY1c18L=ZplMP<~|MP9bT0h^Me$o^TxTnZ0@aJ{3z`Y5&AnOJx(@$VVppARK`=hajsmX8G&VTW&5qw1k{P6<fbdrXleK#rag+7hd9bCG>s`t6>NRH2me~6@R;4XwCa_+C6?_G}U3y%0W9W_$a*_R$y6n5-49lEofxV|S<_1kf}p^<xv)ievgLNY96IAGCN5PHQjrg$Cx+HN?auK#%7lP86)h`qFbvq=sJ<s2&AR8`Rma-=5xe4{l*br?rlcg8%XeQy8FhFk^C!5&Qie9QcN1EbDA5O|^BUE*A*Yk$o7mvH_;cewL)LvIrXf!k)@XmKWsUjO72?M+kmZ-V?L3c3;eEMYgA?>zYV_SfIQn=``s^)5j!3b+4>!^QU*=q`@3V!Q=EG+sO2h15lKm(ed^q|96=x~}TSMv$2wz#9y^@8c=<?j^v#hIiR;gLglD0RBX$`$H?TA;>2Cq2tcGF#INwhoHl8K{kT#GgtyEGPo%VhRdqxl~YmR<2p#*-Dqd`-y)wzt0E~Iu4IzdkDGkB<K5rCpdt1aa{iKns-`gJRYT$~cH}yTyD6VKob7qY+J|#}!}vvb9qL$KBVCeiq97Qo<Su!~%7&sjhkt826br^fkR?*vum-Mb0`0ExJ8k|7;^^uRL`d&)@e1~N+PCnV!f}pr0X_)et<7DdM#?-JeA%EUnfI;us;nM|Ip|oGpKmWpOJ_;LS&ED0^TD`+x|?RP+RJX=4wZ9gbk|mWccN|suOJ@|)ODnys*d`W-AS(ye}RsixK#Y@o*t2vuLn{G-Vk^HSx3h?>gPJr6{&xs^d?sB+TQ->f8Jt=S9<l8qCV+;vh_2b4`TTJB!;q=omr0OD|SaQzWp#|QK2`kZgwZRuK0=QeG9KFxb4ngf5!iQktbJ@zMbpV{W{yt<~;@F>%iL(XY3Q`%Wy-HNJGJCQ`2tS@Eu2A&cH)ayBga0xuUv#x^ed1?0iJ|OD@0Seo^sX@c;47K$q({qmhpAe}C-W0q%$HzRJfK_wc>L1Ub_=l8zR>?%4Yhf$KrLb-o|8tL1)6fid9AieJ$`*ZU9iye#*3bG@woab)n@qCRf>M+5u-eciUFf_>t7U%|hL>%%rZ@!fgsDB(3rJ6hC1_DQ%RANL{rc`2Xy`HJeDq2GvD<m)bkzgyo28M;z(mBA0o{$d8c_$WWWEY<PuzxAA7=6fOVKbrFcufOBX>)hXZ<}XArg653DX+<}#F1ce+KgR)AI9c7j@7|5v$k!D_KH!VI{lfYG<BHPzhOJmftoLud!NhTZ9B1XtIT;aGy}z-#Avklf|9X3CI<n7-r0euXG<HvF%$sX-embp^vJ+(fj!iMO7m6JBWu<7wo8ts4Z@e?s#c#dcMfgn<9E!gKzatY#jb|NvE7CcA3ioC>r0O)OId=V~LLI`&PC&+LyDMdmKjKCP0%aQP75GJdy>F(fJI=h*b|sP#bY}syBgFSf9pQ8~Pjx;z7TASy+^3II8;U`ScVwzbi2^OYujOSG>_JuZt47(cTz`~yknj$R95#sey}9~?4xb-fQ{8!8kIm(QZs^X!u$8kL&VLWL;eVVsTvjB3*54>nXPG~*{-Xn~Xp9p_8m#{Fj!jeC^>faQtmgPGvf(JTr$KjEa7Fj?N`f1MytzwcSW$4OchI;@q;v=9Uw=FtP)812!}%^~SKso(>Hl3A`nz-g5H6f1Rae>KH_rbDQ#|4R=js)MKmURBSH11**7d-;PNB0)Pc`u2n1y~p;I|lV$Ml`>Z)bS+em%i2T`wp3rS(@Q`h8J%ms`gv>+10Q!XTrK{s140`jv4RLqFPNMHU@J?i?;tVm194r@henD+re_%r_mbCB4$(1^}+9eUEk?<hn-Mi9tS7e+^9B57+SV-CJ9c`n$D#n1TC2z9RSl`N5vPD)4VdIBxC>diSh+#PKmy`?6>!E&|^LK>UF1<IwN<{r&h~PXEhYx#GF&h2oo|mjqXc-&Oow>pO9;$#DWLxGG}5V&$8z-IMA~cVD$UtKSj5aPl4oa+)H&kvE0zAf{by_YM_tiF$hw<><BwD>)p!W0OqBOfiadRbuWaXCF+>(UsPpjyiXO4btg1_R3x7zP&H*b~ZmK`MqGQ9QAkY`bfl0;mKdn+_30xpVZPjsvCZXYV^9vFY`NGA+NF{xORlHu1vcfa>uSYto_&1qq}Q&{IO0~)G0*~Umeo#<lq`#T{ZQu58mT|-44brGxY_%8?#+c-8E*5|M8lf_?EyA5MP|g9Q>9^C8HmJT)l_9eQf(qHzFVC{hE+_=GJZ>UQ=V&*zM1|r~Z-}`{2+z03RE^Lf(RqKR-CWFHoNx;JdHwz<=uaiulPP{5`h&^Q&X}Z9?#!{I8AQk9aKObJKTAdCfe2S<cHb`F5i3m-VgR`-LUf^YzYzTiW*9Sssv|o4(@wcm&>&e{BDLmM>#}+4n*6J}UUD5Pod>#Plx*>dx;k?Jwhg?Xft6f8F_l;d?j58U207%gArN6ldVK?f)>&IpN=oa#~-;++dz<Uo+NM7xsS0SpAl1x!u0}Z{)>v!?iT<+oy|<zpe?c$Bo4`Mdj*Nci{Dp+;yVs;9IKn{pR53TLc7N(znkUzPpI{#-n|i-}Qmq(eIT0^)k_|n^ph#<0X^*Lek^L0=R157znwV`bNEDNfOZ)62DEh{WjtDzm|0Sn0Wh`eEZmU&p5tVuaDl(BW?PD-{0g?Ke<d-&OZ3%@)grQ=h<&CcYE~r!=^q^-#C8n1%8SCdOhO~e*E!^t$QH)awm2Z{6FsDm}=ecA%Ef1zd+nQ`X~5HSMc@=_?Y3m%EGPZM$<jH`OFR9U85%FID4GwgtG=KN1OoEaii`?t`w^p>skrNsZvG3FkDB>b&Ujh#cG1<KDcq`Gm_Sv09kjUWfw$d4{o|ZV@bw}xK|M}`|A6=-R9C*m9+G0gnskEoW_5BN&B&syQY^6{0p<cy@&gJS#s3^AMky=ZMqTTc!0OHf4OD)#~Y^W;Gb8Pe}Bc~%KR1RlgIYKb94Lu`ZM;otEv9_Q5w&2v|6qF_?Qj+`**iYH#9DmE1#Z!UU}if?w`2uW&B;*zh&y*&j9fChjMm!^D8Cx;STk#$r8~Ink*Ur&$i;X7k>YrZO3EmH;hN%@AH8F>3!iNh4k(Ae&&?B&KnZ}$Krv}f0ULd=Ey%&`mSaE<?gD?1V3`;nZo^y<?oO5xi?L}A_)BZB@6IRME{o3--+|TyXps?S8@K<(A-YLyRQE0p?LqkDlqrsUw?wpzyEk%>*3t5^uODC7qM%E4?p1VmeBDr?m41sGQja(9gx3%c@K2Icz1tY9jU((Ki?96zZ3w3{^`Dc<w*9&>&4r^8-3S3xlaFTbKa-=F({(w4`<AMl2@o3<_Fe$>PjPBZ{y)X{mb#V?u8-2XXSw7o;<ABh>eIO#fop&^n`=lbfWgF=NWNa8TOjtmW}(CxGC9DIr+ACGp4Tn5wyGQd1OUM(^x5vdAqvVPVoAYV!m5oL3dL*{54;EO|l4v+vaS(=n&u_b-h*QU+=c(A2nh!{y`%`nQt`W&*v_?y3rQY+u_&W+x4%-5uZ)vvmWK$K7OZ~{``fK|9@B$$Nn8ne5=g(53S<2y7YItX2ZH_Ht6#3!Q#K7t_Xi<0ssEwFWyD>FK77Q*HXEw?7fyHoxfEr&%Jrr?%!=(_&p!!mZJS%`iob_37T%aIdS36i8Gz$iz4eLWnI6Fq-j$!MfWc3jU_clT(4Co`oA6?p#yjI-Cobn)Cp=vZVvB}MdjHy5OhV}@%Ut(b@NJB?3|l0yJLDxue)9DSf~Gb)8P)gk>n=ZSVzX$`yGuFQ#;0A6gXB_N%_q&FL%1!LyuoW#BZ;oZ@qUx_&+IJ_%F}f{WSbS_FepL_sz9=%-{&qo5zv8D{p^%6xuHW`(Fs7Lz6q|N|*c)%im0Mt-69A+`jAn2YTR-e?0vEM$OfKQg8LETB|Q-75L4XUdo{EV(}l9NI7eDM>_iH6;ap~|A({pyMiq@Px|LKy5~>#sQ>k&Q5;RGDE3K>-8bxCs-oU1D6eIhe|+#Q?grLHbCXU-CO!b%Z0k!u_~(ZrsYgKi$D>}g=D2Zqqd#gcKGfl2)qz{T3dX(zn-n|a*Gg0{`5hEmZd5Dd&U;bnk8*!{FGq~sorxc4qgoxfjnL1x_@lAj@1s?>F>!<Zm}Gj?0^^)(Sy#9E+h69He*E|`Aa(uC1$HdAqTd76_sTZM$r44~QA?KMN}!{&Or3P?{2lebZd*`%(>H=^#NXWAc3eB_?HVbYR}VmB-HT!8KS6Rl5mu*3mA#6KqhEN%aZ}iPDauXhu$m)6f_;sLIgQm<KV79sGLoQQ(;@epo2@B0ovy=i)x>YK=qT`)(N|=Y<wo?v;p*JD7wYb%fA!$#AvbtmIr4c^__y1ddw}jN_l=cQ#}U3ByOnTrg{~kJd)Mr)Nik*4LCm=I0vnc9A9vyR+;+r08hb_CIC{^S_rdZ1W37D6n%%YmcdU|X8u!!uTcN$8a=d%+kJq#O3xxleuleta$F4U@`ae~U-5|aUyl(Qd^zl2k-3B@srj2S5`%*FZl!|{3Nv%4<nw{ptF-4y+IZS&cCifL}`0aOE;U78s|9Va2!+hNm5$^1{mrBTArtLljV){##`TvIz*ni+~J$yj;RwVz&M^6fOlX{1rZ^2J~72vuhtnA*nIL>JBdl&1gj}`otll2T1Uc!dxHEQ_b3g!>sf7kc<-rf1$#c>vTbT<GSkE(9$QttN$KhD1+=x#mW7ksy-AMez>)cXH}&Akr3js38Y=V-?xcioHM5Oj|yf9vizUTyS0Mde<;v9x);*t!~-e-q(*r0m)!{6^8$KDjl2+V0B0$2wnqLi$k)_<_Oqj(30X^N^d}asTfu;RBjG4{tTpSGWA7_hYHc-SWIR9gm~)8uwovn(IuDa@Fg)eqjt9kMxs3IjuVGNGZ+YH{%VzqjaDhtN5~}$Y%zAj`qJ`L);rRSKPn!_+NPcW#|5h2Vc@@Uvm%l`k61Ie<vRIV|!OsAEE#KTbK`*EgziukDC7@=3naKFYv$9%^yd63>e<E@CP>D9sd_q{P!d9xRLS9-GAZDZ^MlLD%5aYcamq_^tEfGovq>o{`DcV+kNdV!A<FpTb}FP?1%dt2MW0cC0`8ocl)@DMc-qE=YBd9++!-|lj|&N--*GE2K)|!KZ`9Mmue51o;Vy`;7Z#^o%-nge3AJoevhXQuME1;_h8Xy;88hwTbY~4#BYhr4-5Mg0R8b>QNLcs;rdG8jQ<t`DQ|F{6-8VZ6~`!i!g32?e~U%cST`8?_bK`Js|~&z`0qxLcWI_?PR6gr<OYJfk0rRQ(BI|bIrv-oc=Y~y>eqwxe*+NP2L=')).decode("utf-8")
_V138_SOIL_NS = {"__name__": "_moon_v138_soil_v26h", "__file__": "<embedded-soil-v26h>"}
exec(compile(_V138_SOIL_SOURCE, "<embedded-soil-v26h>", "exec"), _V138_SOIL_NS)

_V138_STATE = {0: {}, 1: {}}
_V138_EXTRA_HAND = {
    2: ["PICKUP", "SHEEP", 2],
    3: ["WEST"],
    4: ["NORTH"],
    5: ["NORTH"],
    14: ["PLACE", "SHEEP", 1],
    15: ["WEST"],
    16: ["SOUTH"],
    20: ["PLACE", "SHEEP", 1],
}


def _v138_public_rival_signature(obs):
    seat = _seat(obs)
    farms = list(_get(obs, "farms", []) or [])
    rival = farms[1 - seat] if len(farms) >= 2 else {}
    pastures = 0
    for row in list(_get(rival, "tiles", []) or []):
        for tile in list(row or []):
            kind = str(_get(tile, "kind", tile) or "")
            if kind == "PASTURE":
                pastures += 1
    return (
        len(list(_get(rival, "hands", []) or [])),
        int(_get(rival, "hires_today", 0) or 0),
        pastures,
        int(round(float(_get(rival, "money", 0) or 0))),
    )


def _v138_use_soil(obs):
    hands, hires, pastures, money = _v138_public_rival_signature(obs)
    # These classes cover ten of the seventeen observed Moon V135 public
    # losses.  Frozen Soil won all 20 paired-seat guarded reruns in those
    # classes before the shared-opening compatibility repair was applied.
    # The common two-hand/$280 class is deliberately excluded: it mixes a
    # Soil win with a severe Soil loss and is also common among top agents.
    # Five-hand/$13 (the V92-like population that crushes pure Soil) and
    # five-hand/$19 and $20 remain Moon: $19 collides with current top MiMi
    # in candidate head-to-head state even though replay-only MiMi shows $20.
    return bool(
        (hands == 2 and hires == 2 and pastures == 0 and money == 167)
        or (hands == 4 and hires == 4 and pastures == 0 and money in {4, 7, 71, 76})
        or (hands == 4 and hires == 4 and pastures == 1 and money == 2013)
        or (hands == 5 and hires == 5 and pastures == 0 and money == 3)
        or (hands == 5 and hires == 5 and pastures == 1 and money == 2988)
    )


def _v138_repaired_soil_action(obs, step):
    action = _copy_action(_V138_SOIL_NS["agent"](obs))
    farm = _farm(obs, _seat(obs))
    expected = len(list(_get(farm, "hands", []) or []))
    hands = [list(order or ["PASS"]) for order in (action.get("hands") or [])]
    hands.extend([["PASS"] for _ in range(max(0, expected - len(hands)))])
    action["hands"] = hands[:expected]
    if step == 1:
        action["market"] = [
            ["BUY_ANIMAL", "SHEEP", 2],
            ["BUY_SEED", "WHEAT", 5],
            ["BUY_SEED", "MELON", 5],
        ]
    if 1 <= step <= 23 and expected >= 5:
        action["hands"][4] = list(_V138_EXTRA_HAND.get(step, ["PASS"]))
    return action


def _v138_soil_action(obs, step):
    seat = _seat(obs)
    state = _V138_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {"last_step": step, "active": False}
        _V138_STATE[seat] = state
        # Initialize the embedded policy on the real initial observation.
        _V138_SOIL_NS["agent"](obs)
    state["last_step"] = step
    if step == 1:
        state["active"] = _v138_use_soil(obs)
    if not state.get("active"):
        return None
    return _v138_repaired_soil_action(obs, step)



def agent(obs):
    step = int(_get(obs, "step", 0) or 0)
    soil_action = _v138_soil_action(obs, step)
    if soil_action is not None:
        return soil_action
    return _v138_moon_agent(obs)
