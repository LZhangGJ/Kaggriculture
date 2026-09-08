
/mnt/e/ai_coding/kaggle/kaggriculture/handoff/T1_Rebuild_Divergence_GPT_Review_20260909_v1/binaries/local_rebuilt_gcc13.so:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000019e50 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0>:
   19e50:	41 57                	push   %r15
   19e52:	41 56                	push   %r14
   19e54:	41 55                	push   %r13
   19e56:	41 54                	push   %r12
   19e58:	55                   	push   %rbp
   19e59:	53                   	push   %rbx
   19e5a:	48 81 ec 00 10 00 00 	sub    $0x1000,%rsp
   19e61:	48 83 0c 24 00       	orq    $0x0,(%rsp)
   19e66:	48 81 ec 00 10 00 00 	sub    $0x1000,%rsp
   19e6d:	48 83 0c 24 00       	orq    $0x0,(%rsp)
   19e72:	48 81 ec 58 0c 00 00 	sub    $0xc58,%rsp
   19e79:	66 0f ef c0          	pxor   %xmm0,%xmm0
   19e7d:	48 89 3c 24          	mov    %rdi,(%rsp)
   19e81:	49 89 f4             	mov    %rsi,%r12
   19e84:	41 89 d7             	mov    %edx,%r15d
   19e87:	41 89 cd             	mov    %ecx,%r13d
   19e8a:	45 89 c6             	mov    %r8d,%r14d
   19e8d:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
   19e94:	00 00 
   19e96:	48 89 84 24 48 2c 00 	mov    %rax,0x2c48(%rsp)
   19e9d:	00 
   19e9e:	31 c0                	xor    %eax,%eax
   19ea0:	66 0f 2f 86 80 00 00 	comisd 0x80(%rsi),%xmm0
   19ea7:	00 
   19ea8:	0f 83 ba 03 00 00    	jae    1a268 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x418>
   19eae:	4c 8d 54 24 20       	lea    0x20(%rsp),%r10
   19eb3:	8b ae 8c 34 00 00    	mov    0x348c(%rsi),%ebp
   19eb9:	31 c0                	xor    %eax,%eax
   19ebb:	8d 5a f7             	lea    -0x9(%rdx),%ebx
   19ebe:	4c 89 d7             	mov    %r10,%rdi
   19ec1:	b9 4c 01 00 00       	mov    $0x14c,%ecx
   19ec6:	48 63 db             	movslq %ebx,%rbx
   19ec9:	4c 89 54 24 10       	mov    %r10,0x10(%rsp)
   19ece:	f3 48 ab             	rep stos %rax,%es:(%rdi)
   19ed1:	66 0f ef c9          	pxor   %xmm1,%xmm1
   19ed5:	41 39 ed             	cmp    %ebp,%r13d
   19ed8:	44 89 f7             	mov    %r14d,%edi
   19edb:	48 8d 05 fe 4f 0a 00 	lea    0xa4ffe(%rip),%rax        # beee0 <_ZN3dp7L12animal_priceE>
   19ee2:	41 0f 4d ed          	cmovge %r13d,%ebp
   19ee6:	4c 89 4c 24 18       	mov    %r9,0x18(%rsp)
   19eeb:	f2 0f 2a 0c 98       	cvtsi2sdl (%rax,%rbx,4),%xmm1
   19ef0:	89 94 24 74 0a 00 00 	mov    %edx,0xa74(%rsp)
   19ef7:	c7 84 24 70 0a 00 00 	movl   $0x1e,0xa70(%rsp)
   19efe:	1e 00 00 00 
   19f02:	4c 63 dd             	movslq %ebp,%r11
   19f05:	4c 89 5c 24 08       	mov    %r11,0x8(%rsp)
   19f0a:	f2 0f 11 8c 24 78 0a 	movsd  %xmm1,0xa78(%rsp)
   19f11:	00 00 
   19f13:	f2 42 0f 10 84 dc 80 	movsd  0x980(%rsp,%r11,8),%xmm0
   19f1a:	09 00 00 
   19f1d:	f2 0f 5c c1          	subsd  %xmm1,%xmm0
   19f21:	f2 42 0f 11 84 dc 80 	movsd  %xmm0,0x980(%rsp,%r11,8)
   19f28:	09 00 00 
   19f2b:	e8 90 d1 00 00       	call   270c0 <_ZN3dp74nearEi>
   19f30:	66 0f ef c0          	pxor   %xmm0,%xmm0
   19f34:	4c 8b 5c 24 08       	mov    0x8(%rsp),%r11
   19f39:	31 f6                	xor    %esi,%esi
   19f3b:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
   19f3f:	f2 0f 59 05 99 5a 0a 	mulsd  0xa5a99(%rip),%xmm0        # bf9e0 <_ZZNSt19_Sp_make_shared_tag5_S_tiEvE5__tag+0x118>
   19f46:	00 
   19f47:	ba c0 21 00 00       	mov    $0x21c0,%edx
   19f4c:	f2 0f 10 1d 94 5a 0a 	movsd  0xa5a94(%rip),%xmm3        # bf9e8 <_ZZNSt19_Sp_make_shared_tag5_S_tiEvE5__tag+0x120>
   19f53:	00 
   19f54:	48 8d bc 24 80 0a 00 	lea    0xa80(%rsp),%rdi
   19f5b:	00 
   19f5c:	f2 0f 58 c3          	addsd  %xmm3,%xmm0
   19f60:	f2 42 0f 58 84 dc 90 	addsd  0x890(%rsp,%r11,8),%xmm0
   19f67:	08 00 00 
   19f6a:	f2 42 0f 11 84 dc 90 	movsd  %xmm0,0x890(%rsp,%r11,8)
   19f71:	08 00 00 
   19f74:	e8 c7 a4 ff ff       	call   14440 <memset@plt>
   19f79:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
   19f7e:	89 e9                	mov    %ebp,%ecx
   19f80:	44 89 ea             	mov    %r13d,%edx
   19f83:	f2 41 0f 10 44 24 28 	movsd  0x28(%r12),%xmm0
   19f8a:	48 89 c7             	mov    %rax,%rdi
   19f8d:	44 89 fe             	mov    %r15d,%esi
   19f90:	e8 ab 73 01 00       	call   31340 <_ZN11competitive15AnimalServiceDP5solveEiiiRKSt5arrayIS1_IdLm9EELm30EEd>
   19f95:	83 fd 1c             	cmp    $0x1c,%ebp
   19f98:	4c 8b 54 24 10       	mov    0x10(%rsp),%r10
   19f9d:	0f 8f 82 02 00 00    	jg     1a225 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x3d5>
   19fa3:	49 63 c6             	movslq %r14d,%rax
   19fa6:	44 89 f2             	mov    %r14d,%edx
   19fa9:	4c 8b 5c 24 08       	mov    0x8(%rsp),%r11
   19fae:	66 0f ef d2          	pxor   %xmm2,%xmm2
   19fb2:	48 69 c0 67 66 66 66 	imul   $0x66666667,%rax,%rax
   19fb9:	c1 fa 1f             	sar    $0x1f,%edx
   19fbc:	f2 41 0f 10 64 24 30 	movsd  0x30(%r12),%xmm4
   19fc3:	4c 89 54 24 08       	mov    %r10,0x8(%rsp)
   19fc8:	f2 0f 10 0d 90 5d 0a 	movsd  0xa5d90(%rip),%xmm1        # bfd60 <_ZZNSt19_Sp_make_shared_tag5_S_tiEvE5__tag+0x498>
   19fcf:	00 
   19fd0:	f2 0f 10 1d 10 5a 0a 	movsd  0xa5a10(%rip),%xmm3        # bf9e8 <_ZZNSt19_Sp_make_shared_tag5_S_tiEvE5__tag+0x120>
   19fd7:	00 
   19fd8:	44 8d 7d 01          	lea    0x1(%rbp),%r15d
   19fdc:	4c 8d 25 dd 4e 0a 00 	lea    0xa4edd(%rip),%r12        # beec0 <_ZN3dp7L9aintervalE>
   19fe3:	48 c1 f8 22          	sar    $0x22,%rax
   19fe7:	29 d0                	sub    %edx,%eax
   19fe9:	8d 14 80             	lea    (%rax,%rax,4),%edx
   19fec:	89 c7                	mov    %eax,%edi
   19fee:	44 89 f0             	mov    %r14d,%eax
   19ff1:	01 d2                	add    %edx,%edx
   19ff3:	4c 8d 35 b6 4e 0a 00 	lea    0xa4eb6(%rip),%r14        # beeb0 <_ZN3dp7L4heldE>
   19ffa:	29 d0                	sub    %edx,%eax
   19ffc:	8d 50 fc             	lea    -0x4(%rax),%edx
   19fff:	89 d1                	mov    %edx,%ecx
   1a001:	f7 d9                	neg    %ecx
   1a003:	0f 48 ca             	cmovs  %edx,%ecx
   1a006:	8d 57 fc             	lea    -0x4(%rdi),%edx
   1a009:	89 d6                	mov    %edx,%esi
   1a00b:	f7 de                	neg    %esi
   1a00d:	0f 48 f2             	cmovs  %edx,%esi
   1a010:	83 e8 05             	sub    $0x5,%eax
   1a013:	89 c2                	mov    %eax,%edx
   1a015:	f7 da                	neg    %edx
   1a017:	0f 48 d0             	cmovs  %eax,%edx
   1a01a:	8d 47 fb             	lea    -0x5(%rdi),%eax
   1a01d:	89 c7                	mov    %eax,%edi
   1a01f:	f7 df                	neg    %edi
   1a021:	0f 48 f8             	cmovs  %eax,%edi
   1a024:	8d 04 32             	lea    (%rdx,%rsi,1),%eax
   1a027:	01 ce                	add    %ecx,%esi
   1a029:	39 f0                	cmp    %esi,%eax
   1a02b:	0f 4f c6             	cmovg  %esi,%eax
   1a02e:	be 64 00 00 00       	mov    $0x64,%esi
   1a033:	39 f0                	cmp    %esi,%eax
   1a035:	0f 4f c6             	cmovg  %esi,%eax
   1a038:	01 f9                	add    %edi,%ecx
   1a03a:	39 c8                	cmp    %ecx,%eax
   1a03c:	0f 4f c1             	cmovg  %ecx,%eax
   1a03f:	01 fa                	add    %edi,%edx
   1a041:	4a 8d 0c dd 00 00 00 	lea    0x0(,%r11,8),%rcx
   1a048:	00 
   1a049:	39 d0                	cmp    %edx,%eax
   1a04b:	0f 4f c2             	cmovg  %edx,%eax
   1a04e:	49 01 cb             	add    %rcx,%r11
   1a051:	45 31 c9             	xor    %r9d,%r9d
   1a054:	4c 01 d1             	add    %r10,%rcx
   1a057:	4f 8d 04 da          	lea    (%r10,%r11,8),%r8
   1a05b:	45 31 db             	xor    %r11d,%r11d
   1a05e:	f2 0f 2a d0          	cvtsi2sd %eax,%xmm2
   1a062:	f2 0f 59 15 26 59 0a 	mulsd  0xa5926(%rip),%xmm2        # bf990 <_ZZNSt19_Sp_make_shared_tag5_S_tiEvE5__tag+0xc8>
   1a069:	00 
   1a06a:	89 e8                	mov    %ebp,%eax
   1a06c:	44 29 e8             	sub    %r13d,%eax
   1a06f:	4c 8d 2d 5a 4e 0a 00 	lea    0xa4e5a(%rip),%r13        # beed0 <_ZN3dp7L6afirstE>
   1a076:	8d 78 01             	lea    0x1(%rax),%edi
   1a079:	e9 e1 00 00 00       	jmp    1a15f <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x30f>
   1a07e:	66 90                	xchg   %ax,%ax
   1a080:	f2 41 0f 10 00       	movsd  (%r8),%xmm0
   1a085:	41 8b 54 9d 00       	mov    0x0(%r13,%rbx,4),%edx
   1a08a:	45 89 fa             	mov    %r15d,%r10d
   1a08d:	f2 0f 5c c1          	subsd  %xmm1,%xmm0
   1a091:	f2 41 0f 11 00       	movsd  %xmm0,(%r8)
   1a096:	66 0f 28 c2          	movapd %xmm2,%xmm0
   1a09a:	f2 0f 58 c3          	addsd  %xmm3,%xmm0
   1a09e:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
   1a0a2:	f2 0f 58 81 70 08 00 	addsd  0x870(%rcx),%xmm0
   1a0a9:	00 
   1a0aa:	f2 0f 11 81 70 08 00 	movsd  %xmm0,0x870(%rcx)
   1a0b1:	00 
   1a0b2:	39 d7                	cmp    %edx,%edi
   1a0b4:	0f 8c 06 01 00 00    	jl     1a1c0 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x370>
   1a0ba:	89 f8                	mov    %edi,%eax
   1a0bc:	49 63 f7             	movslq %r15d,%rsi
   1a0bf:	29 d0                	sub    %edx,%eax
   1a0c1:	99                   	cltd
   1a0c2:	41 f7 3c 9c          	idivl  (%r12,%rbx,4)
   1a0c6:	48 8d 04 f5 00 00 00 	lea    0x0(,%rsi,8),%rax
   1a0cd:	00 
   1a0ce:	85 d2                	test   %edx,%edx
   1a0d0:	75 4c                	jne    1a11e <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x2ce>
   1a0d2:	48 8d 05 c7 4d 0a 00 	lea    0xa4dc7(%rip),%rax        # beea0 <_ZN3dp7L7productE>
   1a0d9:	41 83 c1 01          	add    $0x1,%r9d
   1a0dd:	66 0f ef c0          	pxor   %xmm0,%xmm0
   1a0e1:	48 63 14 98          	movslq (%rax,%rbx,4),%rdx
   1a0e5:	48 8d 04 f5 00 00 00 	lea    0x0(,%rsi,8),%rax
   1a0ec:	00 
   1a0ed:	f2 41 0f 2a c1       	cvtsi2sd %r9d,%xmm0
   1a0f2:	4c 8d 1c 30          	lea    (%rax,%rsi,1),%r11
   1a0f6:	45 31 c9             	xor    %r9d,%r9d
   1a0f9:	4c 01 da             	add    %r11,%rdx
   1a0fc:	f2 0f 58 44 d4 20    	addsd  0x20(%rsp,%rdx,8),%xmm0
   1a102:	f2 0f 11 44 d4 20    	movsd  %xmm0,0x20(%rsp,%rdx,8)
   1a108:	f2 0f 10 84 f4 90 08 	movsd  0x890(%rsp,%rsi,8),%xmm0
   1a10f:	00 00 
   1a111:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
   1a115:	f2 0f 11 84 f4 90 08 	movsd  %xmm0,0x890(%rsp,%rsi,8)
   1a11c:	00 00 
   1a11e:	41 8b 14 9e          	mov    (%r14,%rbx,4),%edx
   1a122:	41 83 c1 01          	add    $0x1,%r9d
   1a126:	83 ea 01             	sub    $0x1,%edx
   1a129:	44 39 ca             	cmp    %r9d,%edx
   1a12c:	44 0f 4e ca          	cmovle %edx,%r9d
   1a130:	45 31 db             	xor    %r11d,%r11d
   1a133:	48 01 f0             	add    %rsi,%rax
   1a136:	41 83 c7 01          	add    $0x1,%r15d
   1a13a:	83 c7 01             	add    $0x1,%edi
   1a13d:	f2 0f 10 44 c4 60    	movsd  0x60(%rsp,%rax,8),%xmm0
   1a143:	49 83 c0 48          	add    $0x48,%r8
   1a147:	48 83 c1 08          	add    $0x8,%rcx
   1a14b:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
   1a14f:	f2 0f 11 44 c4 60    	movsd  %xmm0,0x60(%rsp,%rax,8)
   1a155:	41 83 fa 1d          	cmp    $0x1d,%r10d
   1a159:	0f 84 c1 00 00 00    	je     1a220 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x3d0>
   1a15f:	41 8d 47 ff          	lea    -0x1(%r15),%eax
   1a163:	39 c5                	cmp    %eax,%ebp
   1a165:	0f 84 15 ff ff ff    	je     1a080 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x230>
   1a16b:	66 0f 28 c2          	movapd %xmm2,%xmm0
   1a16f:	41 83 c3 01          	add    $0x1,%r11d
   1a173:	45 89 fa             	mov    %r15d,%r10d
   1a176:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
   1a17a:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
   1a17e:	f2 0f 58 81 70 08 00 	addsd  0x870(%rcx),%xmm0
   1a185:	00 
   1a186:	f2 0f 11 81 70 08 00 	movsd  %xmm0,0x870(%rcx)
   1a18d:	00 
   1a18e:	41 83 fb 02          	cmp    $0x2,%r11d
   1a192:	0f 84 42 02 00 00    	je     1a3da <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x58a>
   1a198:	41 8b 54 9d 00       	mov    0x0(%r13,%rbx,4),%edx
   1a19d:	49 63 f7             	movslq %r15d,%rsi
   1a1a0:	39 d7                	cmp    %edx,%edi
   1a1a2:	7c 0d                	jl     1a1b1 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x361>
   1a1a4:	89 f8                	mov    %edi,%eax
   1a1a6:	29 d0                	sub    %edx,%eax
   1a1a8:	99                   	cltd
   1a1a9:	41 f7 3c 9c          	idivl  (%r12,%rbx,4)
   1a1ad:	85 d2                	test   %edx,%edx
   1a1af:	74 1f                	je     1a1d0 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x380>
   1a1b1:	48 8d 04 f5 00 00 00 	lea    0x0(,%rsi,8),%rax
   1a1b8:	00 
   1a1b9:	e9 75 ff ff ff       	jmp    1a133 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x2e3>
   1a1be:	66 90                	xchg   %ax,%ax
   1a1c0:	49 63 f7             	movslq %r15d,%rsi
   1a1c3:	48 8d 04 f5 00 00 00 	lea    0x0(,%rsi,8),%rax
   1a1ca:	00 
   1a1cb:	e9 4e ff ff ff       	jmp    1a11e <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x2ce>
   1a1d0:	48 8d 05 c9 4c 0a 00 	lea    0xa4cc9(%rip),%rax        # beea0 <_ZN3dp7L7productE>
   1a1d7:	48 63 14 98          	movslq (%rax,%rbx,4),%rdx
   1a1db:	48 8d 04 f5 00 00 00 	lea    0x0(,%rsi,8),%rax
   1a1e2:	00 
   1a1e3:	4c 8d 0c 30          	lea    (%rax,%rsi,1),%r9
   1a1e7:	4c 01 ca             	add    %r9,%rdx
   1a1ea:	45 31 c9             	xor    %r9d,%r9d
   1a1ed:	f2 0f 10 44 d4 20    	movsd  0x20(%rsp,%rdx,8),%xmm0
   1a1f3:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
   1a1f7:	f2 0f 11 44 d4 20    	movsd  %xmm0,0x20(%rsp,%rdx,8)
   1a1fd:	f2 0f 10 84 f4 90 08 	movsd  0x890(%rsp,%rsi,8),%xmm0
   1a204:	00 00 
   1a206:	f2 0f 58 c1          	addsd  %xmm1,%xmm0
   1a20a:	f2 0f 11 84 f4 90 08 	movsd  %xmm0,0x890(%rsp,%rsi,8)
   1a211:	00 00 
   1a213:	e9 1b ff ff ff       	jmp    1a133 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x2e3>
   1a218:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
   1a21f:	00 
   1a220:	4c 8b 54 24 08       	mov    0x8(%rsp),%r10
   1a225:	48 8b 3c 24          	mov    (%rsp),%rdi
   1a229:	b9 4c 01 00 00       	mov    $0x14c,%ecx
   1a22e:	4c 89 d6             	mov    %r10,%rsi
   1a231:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
   1a234:	48 8b 84 24 48 2c 00 	mov    0x2c48(%rsp),%rax
   1a23b:	00 
   1a23c:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
   1a243:	00 00 
   1a245:	0f 85 a1 01 00 00    	jne    1a3ec <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x59c>
   1a24b:	48 8b 04 24          	mov    (%rsp),%rax
   1a24f:	48 81 c4 58 2c 00 00 	add    $0x2c58,%rsp
   1a256:	5b                   	pop    %rbx
   1a257:	5d                   	pop    %rbp
   1a258:	41 5c                	pop    %r12
   1a25a:	41 5d                	pop    %r13
   1a25c:	41 5e                	pop    %r14
   1a25e:	41 5f                	pop    %r15
   1a260:	c3                   	ret
   1a261:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
   1a268:	48 8d 9c 24 80 0a 00 	lea    0xa80(%rsp),%rbx
   1a26f:	00 
   1a270:	48 89 fd             	mov    %rdi,%rbp
   1a273:	48 8d b6 00 01 00 00 	lea    0x100(%rsi),%rsi
   1a27a:	45 31 c9             	xor    %r9d,%r9d
   1a27d:	48 89 df             	mov    %rbx,%rdi
   1a280:	e8 3b 63 01 00       	call   305c0 <_ZNK11competitive7Planner13animal_streamEiiiPKN7fastkag4TileE>
   1a285:	b9 4c 01 00 00       	mov    $0x14c,%ecx
   1a28a:	48 89 ef             	mov    %rbp,%rdi
   1a28d:	48 89 de             	mov    %rbx,%rsi
   1a290:	66 0f 28 8c 24 f0 12 	movapd 0x12f0(%rsp),%xmm1
   1a297:	00 00 
   1a299:	f2 41 0f 10 44 24 30 	movsd  0x30(%r12),%xmm0
   1a2a0:	66 0f 14 c0          	unpcklpd %xmm0,%xmm0
   1a2a4:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1a2a8:	0f 29 8c 24 f0 12 00 	movaps %xmm1,0x12f0(%rsp)
   1a2af:	00 
   1a2b0:	66 0f 28 8c 24 00 13 	movapd 0x1300(%rsp),%xmm1
   1a2b7:	00 00 
   1a2b9:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1a2bd:	0f 29 8c 24 00 13 00 	movaps %xmm1,0x1300(%rsp)
   1a2c4:	00 
   1a2c5:	66 0f 28 8c 24 10 13 	movapd 0x1310(%rsp),%xmm1
   1a2cc:	00 00 
   1a2ce:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1a2d2:	0f 29 8c 24 10 13 00 	movaps %xmm1,0x1310(%rsp)
   1a2d9:	00 
   1a2da:	66 0f 28 8c 24 20 13 	movapd 0x1320(%rsp),%xmm1
   1a2e1:	00 00 
   1a2e3:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1a2e7:	0f 29 8c 24 20 13 00 	movaps %xmm1,0x1320(%rsp)
   1a2ee:	00 
   1a2ef:	66 0f 28 8c 24 30 13 	movapd 0x1330(%rsp),%xmm1
   1a2f6:	00 00 
   1a2f8:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1a2fc:	0f 29 8c 24 30 13 00 	movaps %xmm1,0x1330(%rsp)
   1a303:	00 
   1a304:	66 0f 28 8c 24 40 13 	movapd 0x1340(%rsp),%xmm1
   1a30b:	00 00 
   1a30d:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1a311:	0f 29 8c 24 40 13 00 	movaps %xmm1,0x1340(%rsp)
   1a318:	00 
   1a319:	66 0f 28 8c 24 50 13 	movapd 0x1350(%rsp),%xmm1
   1a320:	00 00 
   1a322:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1a326:	0f 29 8c 24 50 13 00 	movaps %xmm1,0x1350(%rsp)
   1a32d:	00 
   1a32e:	66 0f 28 8c 24 60 13 	movapd 0x1360(%rsp),%xmm1
   1a335:	00 00 
   1a337:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1a33b:	0f 29 8c 24 60 13 00 	movaps %xmm1,0x1360(%rsp)
   1a342:	00 
   1a343:	66 0f 28 8c 24 70 13 	movapd 0x1370(%rsp),%xmm1
   1a34a:	00 00 
   1a34c:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1a350:	0f 29 8c 24 70 13 00 	movaps %xmm1,0x1370(%rsp)
   1a357:	00 
   1a358:	66 0f 28 8c 24 80 13 	movapd 0x1380(%rsp),%xmm1
   1a35f:	00 00 
   1a361:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1a365:	0f 29 8c 24 80 13 00 	movaps %xmm1,0x1380(%rsp)
   1a36c:	00 
   1a36d:	66 0f 28 8c 24 90 13 	movapd 0x1390(%rsp),%xmm1
   1a374:	00 00 
   1a376:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1a37a:	0f 29 8c 24 90 13 00 	movaps %xmm1,0x1390(%rsp)
   1a381:	00 
   1a382:	66 0f 28 8c 24 a0 13 	movapd 0x13a0(%rsp),%xmm1
   1a389:	00 00 
   1a38b:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1a38f:	0f 29 8c 24 a0 13 00 	movaps %xmm1,0x13a0(%rsp)
   1a396:	00 
   1a397:	66 0f 28 8c 24 b0 13 	movapd 0x13b0(%rsp),%xmm1
   1a39e:	00 00 
   1a3a0:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1a3a4:	0f 29 8c 24 b0 13 00 	movaps %xmm1,0x13b0(%rsp)
   1a3ab:	00 
   1a3ac:	66 0f 28 8c 24 c0 13 	movapd 0x13c0(%rsp),%xmm1
   1a3b3:	00 00 
   1a3b5:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1a3b9:	66 0f 59 84 24 d0 13 	mulpd  0x13d0(%rsp),%xmm0
   1a3c0:	00 00 
   1a3c2:	0f 29 8c 24 c0 13 00 	movaps %xmm1,0x13c0(%rsp)
   1a3c9:	00 
   1a3ca:	0f 29 84 24 d0 13 00 	movaps %xmm0,0x13d0(%rsp)
   1a3d1:	00 
   1a3d2:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
   1a3d5:	e9 5a fe ff ff       	jmp    1a234 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x3e4>
   1a3da:	44 89 bc 24 70 0a 00 	mov    %r15d,0xa70(%rsp)
   1a3e1:	00 
   1a3e2:	4c 8b 54 24 08       	mov    0x8(%rsp),%r10
   1a3e7:	e9 39 fe ff ff       	jmp    1a225 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x3d5>
   1a3ec:	e8 2f a1 ff ff       	call   14520 <__stack_chk_fail@plt>

Disassembly of section .fini:
