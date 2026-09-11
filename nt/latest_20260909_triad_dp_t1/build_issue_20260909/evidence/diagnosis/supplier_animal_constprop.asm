
/mnt/e/ai_coding/kaggle/kaggriculture/handoff/T1_Rebuild_Divergence_GPT_Review_20260909_v1/binaries/supplier_original.so:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .text:

0000000000018990 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0>:
   18990:	41 57                	push   %r15
   18992:	66 0f ef c0          	pxor   %xmm0,%xmm0
   18996:	41 89 cb             	mov    %ecx,%r11d
   18999:	45 89 c7             	mov    %r8d,%r15d
   1899c:	41 56                	push   %r14
   1899e:	49 89 fe             	mov    %rdi,%r14
   189a1:	41 55                	push   %r13
   189a3:	49 89 f5             	mov    %rsi,%r13
   189a6:	41 54                	push   %r12
   189a8:	55                   	push   %rbp
   189a9:	53                   	push   %rbx
   189aa:	48 81 ec 88 2c 00 00 	sub    $0x2c88,%rsp
   189b1:	66 0f 2f 86 80 00 00 	comisd 0x80(%rsi),%xmm0
   189b8:	00 
   189b9:	4c 89 4c 24 18       	mov    %r9,0x18(%rsp)
   189be:	0f 83 87 07 00 00    	jae    1914b <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x7bb>
   189c4:	31 c0                	xor    %eax,%eax
   189c6:	48 8d 7c 24 60       	lea    0x60(%rsp),%rdi
   189cb:	66 0f ef c9          	pxor   %xmm1,%xmm1
   189cf:	b9 4c 01 00 00       	mov    $0x14c,%ecx
   189d4:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
   189d9:	f3 48 ab             	rep stos %rax,%es:(%rdi)
   189dc:	8b 86 8c 34 00 00    	mov    0x348c(%rsi),%eax
   189e2:	44 89 c7             	mov    %r8d,%edi
   189e5:	89 94 24 b4 0a 00 00 	mov    %edx,0xab4(%rsp)
   189ec:	c7 84 24 b0 0a 00 00 	movl   $0x1e,0xab0(%rsp)
   189f3:	1e 00 00 00 
   189f7:	41 39 c3             	cmp    %eax,%r11d
   189fa:	89 c3                	mov    %eax,%ebx
   189fc:	8d 42 f7             	lea    -0x9(%rdx),%eax
   189ff:	44 89 5c 24 20       	mov    %r11d,0x20(%rsp)
   18a04:	48 63 e8             	movslq %eax,%rbp
   18a07:	48 8d 05 92 55 0b 00 	lea    0xb5592(%rip),%rax        # cdfa0 <_ZN3dp7L12animal_priceE>
   18a0e:	41 0f 4d db          	cmovge %r11d,%ebx
   18a12:	f2 0f 2a 0c a8       	cvtsi2sdl (%rax,%rbp,4),%xmm1
   18a17:	48 89 6c 24 10       	mov    %rbp,0x10(%rsp)
   18a1c:	4c 63 cb             	movslq %ebx,%r9
   18a1f:	89 5c 24 3c          	mov    %ebx,0x3c(%rsp)
   18a23:	4c 89 0c 24          	mov    %r9,(%rsp)
   18a27:	f2 0f 11 8c 24 b8 0a 	movsd  %xmm1,0xab8(%rsp)
   18a2e:	00 00 
   18a30:	f2 42 0f 10 84 cc c0 	movsd  0x9c0(%rsp,%r9,8),%xmm0
   18a37:	09 00 00 
   18a3a:	f2 0f 5c c1          	subsd  %xmm1,%xmm0
   18a3e:	f2 42 0f 11 84 cc c0 	movsd  %xmm0,0x9c0(%rsp,%r9,8)
   18a45:	09 00 00 
   18a48:	e8 73 d9 00 00       	call   263c0 <_ZN3dp74nearEi>
   18a4d:	66 0f ef c0          	pxor   %xmm0,%xmm0
   18a51:	4c 8b 0c 24          	mov    (%rsp),%r9
   18a55:	31 f6                	xor    %esi,%esi
   18a57:	f2 0f 2a c0          	cvtsi2sd %eax,%xmm0
   18a5b:	f2 0f 59 05 2d 57 0b 	mulsd  0xb572d(%rip),%xmm0        # ce190 <C.449.8+0x108>
   18a62:	00 
   18a63:	ba c0 21 00 00       	mov    $0x21c0,%edx
   18a68:	f2 44 0f 10 35 27 57 	movsd  0xb5727(%rip),%xmm14        # ce198 <C.449.8+0x110>
   18a6f:	0b 00 
   18a71:	48 8d 84 24 c0 0a 00 	lea    0xac0(%rsp),%rax
   18a78:	00 
   18a79:	4c 89 4c 24 08       	mov    %r9,0x8(%rsp)
   18a7e:	48 89 c7             	mov    %rax,%rdi
   18a81:	48 89 04 24          	mov    %rax,(%rsp)
   18a85:	f2 41 0f 58 c6       	addsd  %xmm14,%xmm0
   18a8a:	f2 42 0f 58 84 cc d0 	addsd  0x8d0(%rsp,%r9,8),%xmm0
   18a91:	08 00 00 
   18a94:	f2 42 0f 11 84 cc d0 	movsd  %xmm0,0x8d0(%rsp,%r9,8)
   18a9b:	08 00 00 
   18a9e:	e8 6d a6 ff ff       	call   13110 <memset@plt>
   18aa3:	48 8d 05 c6 54 0b 00 	lea    0xb54c6(%rip),%rax        # cdf70 <_ZN3dp7L4heldE>
   18aaa:	f2 41 0f 10 6d 28    	movsd  0x28(%r13),%xmm5
   18ab0:	8b 04 a8             	mov    (%rax,%rbp,4),%eax
   18ab3:	89 44 24 24          	mov    %eax,0x24(%rsp)
   18ab7:	8d 70 ff             	lea    -0x1(%rax),%esi
   18aba:	83 fb 1c             	cmp    $0x1c,%ebx
   18abd:	0f 8f 63 06 00 00    	jg     19126 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x796>
   18ac3:	44 8b 5c 24 20       	mov    0x20(%rsp),%r11d
   18ac8:	4c 8b 4c 24 08       	mov    0x8(%rsp),%r9
   18acd:	41 89 da             	mov    %ebx,%r10d
   18ad0:	4c 89 74 24 48       	mov    %r14,0x48(%rsp)
   18ad5:	44 89 54 24 58       	mov    %r10d,0x58(%rsp)
   18ada:	b9 80 0a 00 00       	mov    $0xa80,%ecx
   18adf:	66 45 0f ef c0       	pxor   %xmm8,%xmm8
   18ae4:	48 8d 05 a5 54 0b 00 	lea    0xb54a5(%rip),%rax        # cdf90 <_ZN3dp7L6afirstE>
   18aeb:	44 8b 24 a8          	mov    (%rax,%rbp,4),%r12d
   18aef:	b8 1d 00 00 00       	mov    $0x1d,%eax
   18af4:	4c 89 6c 24 50       	mov    %r13,0x50(%rsp)
   18af9:	44 29 d8             	sub    %r11d,%eax
   18afc:	4c 89 4c 24 40       	mov    %r9,0x40(%rsp)
   18b01:	f3 44 0f 7e 2d d6 58 	movq   0xb58d6(%rip),%xmm13        # ce3e0 <C.1461.14+0x30>
   18b08:	0b 00 
   18b0a:	89 c2                	mov    %eax,%edx
   18b0c:	b8 1c 00 00 00       	mov    $0x1c,%eax
   18b11:	4c 8b 0c 24          	mov    (%rsp),%r9
   18b15:	44 89 5c 24 5c       	mov    %r11d,0x5c(%rsp)
   18b1a:	44 29 e2             	sub    %r12d,%edx
   18b1d:	29 d8                	sub    %ebx,%eax
   18b1f:	45 89 e0             	mov    %r12d,%r8d
   18b22:	bb 1d 00 00 00       	mov    $0x1d,%ebx
   18b27:	f2 44 0f 10 25 78 56 	movsd  0xb5678(%rip),%xmm12        # ce1a8 <C.449.8+0x120>
   18b2e:	0b 00 
   18b30:	89 d7                	mov    %edx,%edi
   18b32:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
   18b37:	49 89 cb             	mov    %rcx,%r11
   18b3a:	f2 44 0f 10 35 55 56 	movsd  0xb5655(%rip),%xmm14        # ce198 <C.449.8+0x110>
   18b41:	0b 00 
   18b43:	48 8d aa e0 07 00 00 	lea    0x7e0(%rdx),%rbp
   18b4a:	ba 1c 00 00 00       	mov    $0x1c,%edx
   18b4f:	48 29 c2             	sub    %rax,%rdx
   18b52:	48 89 54 24 28       	mov    %rdx,0x28(%rsp)
   18b57:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
   18b5e:	00 00 
   18b60:	41 8d 04 38          	lea    (%r8,%rdi,1),%eax
   18b64:	89 d9                	mov    %ebx,%ecx
   18b66:	45 31 e4             	xor    %r12d,%r12d
   18b69:	41 39 c0             	cmp    %eax,%r8d
   18b6c:	7f 19                	jg     18b87 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x1f7>
   18b6e:	89 f8                	mov    %edi,%eax
   18b70:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
   18b75:	4c 8d 15 04 54 0b 00 	lea    0xb5404(%rip),%r10        # cdf80 <_ZN3dp7L9aintervalE>
   18b7c:	99                   	cltd
   18b7d:	43 f7 3c b2          	idivl  (%r10,%r14,4)
   18b81:	85 d2                	test   %edx,%edx
   18b83:	41 0f 94 c4          	sete   %r12b
   18b87:	4f 8d 34 19          	lea    (%r9,%r11,1),%r14
   18b8b:	66 45 0f ef db       	pxor   %xmm11,%xmm11
   18b90:	45 84 e4             	test   %r12b,%r12b
   18b93:	74 09                	je     18b9e <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x20e>
   18b95:	f2 44 0f 10 1d fa 58 	movsd  0xb58fa(%rip),%xmm11        # ce498 <C.1461.14+0xe8>
   18b9c:	0b 00 
   18b9e:	45 31 ed             	xor    %r13d,%r13d
   18ba1:	85 f6                	test   %esi,%esi
   18ba3:	0f 88 27 02 00 00    	js     18dd0 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x440>
   18ba9:	89 4c 24 20          	mov    %ecx,0x20(%rsp)
   18bad:	4c 89 d8             	mov    %r11,%rax
   18bb0:	44 89 44 24 38       	mov    %r8d,0x38(%rsp)
   18bb5:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
   18bba:	48 8d 15 9f 53 0b 00 	lea    0xb539f(%rip),%rdx        # cdf60 <_ZN3dp7L7productE>
   18bc1:	f2 44 0f 10 55 00    	movsd  0x0(%rbp),%xmm10
   18bc7:	66 44 0f 28 cd       	movapd %xmm5,%xmm9
   18bcc:	f2 0f 10 95 88 00 00 	movsd  0x88(%rbp),%xmm2
   18bd3:	00 
   18bd4:	f2 45 0f 59 c8       	mulsd  %xmm8,%xmm9
   18bd9:	4f 8d 04 36          	lea    (%r14,%r14,1),%r8
   18bdd:	48 89 2c 24          	mov    %rbp,(%rsp)
   18be1:	48 63 0c 8a          	movslq (%rdx,%rcx,4),%rcx
   18be5:	f2 44 0f 58 d5       	addsd  %xmm5,%xmm10
   18bea:	48 8d 14 db          	lea    (%rbx,%rbx,8),%rdx
   18bee:	66 41 0f 28 c0       	movapd %xmm8,%xmm0
   18bf3:	f2 0f 5c d5          	subsd  %xmm5,%xmm2
   18bf7:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
   18bfc:	4c 63 5c 24 24       	movslq 0x24(%rsp),%r11
   18c01:	4d 29 c8             	sub    %r9,%r8
   18c04:	48 01 ca             	add    %rcx,%rdx
   18c07:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
   18c0c:	f2 0f 10 25 8c 55 0b 	movsd  0xb558c(%rip),%xmm4        # ce1a0 <C.449.8+0x118>
   18c13:	00 
   18c14:	66 41 0f 28 f2       	movapd %xmm10,%xmm6
   18c19:	f2 0f 10 1c d1       	movsd  (%rcx,%rdx,8),%xmm3
   18c1e:	f2 0f c2 c2 01       	cmpltsd %xmm2,%xmm0
   18c23:	48 63 4c 24 20       	movslq 0x20(%rsp),%rcx
   18c28:	31 d2                	xor    %edx,%edx
   18c2a:	66 41 0f 57 f5       	xorpd  %xmm13,%xmm6
   18c2f:	66 0f 28 fe          	movapd %xmm6,%xmm7
   18c33:	4c 8d 14 49          	lea    (%rcx,%rcx,2),%r10
   18c37:	f2 41 0f 5c f9       	subsd  %xmm9,%xmm7
   18c3c:	66 0f 54 d0          	andpd  %xmm0,%xmm2
   18c40:	49 c1 e2 02          	shl    $0x2,%r10
   18c44:	eb 36                	jmp    18c7c <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x2ec>
   18c46:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
   18c4d:	00 00 00 
   18c50:	b9 01 00 00 00       	mov    $0x1,%ecx
   18c55:	bb 01 00 00 00       	mov    $0x1,%ebx
   18c5a:	41 89 18             	mov    %ebx,(%r8)
   18c5d:	49 83 c0 10          	add    $0x10,%r8
   18c61:	41 89 48 f4          	mov    %ecx,-0xc(%r8)
   18c65:	f2 41 0f 11 8c d6 80 	movsd  %xmm1,0x1680(%r14,%rdx,8)
   18c6c:	16 00 00 
   18c6f:	48 83 c2 01          	add    $0x1,%rdx
   18c73:	49 39 d3             	cmp    %rdx,%r11
   18c76:	0f 84 24 01 00 00    	je     18da0 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x410>
   18c7c:	89 d1                	mov    %edx,%ecx
   18c7e:	89 d3                	mov    %edx,%ebx
   18c80:	66 0f ef c9          	pxor   %xmm1,%xmm1
   18c84:	41 83 fd 01          	cmp    $0x1,%r13d
   18c88:	74 47                	je     18cd1 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x341>
   18c8a:	66 41 0f 28 ca       	movapd %xmm10,%xmm1
   18c8f:	66 41 0f 28 c3       	movapd %xmm11,%xmm0
   18c94:	45 84 e4             	test   %r12b,%r12b
   18c97:	89 d5                	mov    %edx,%ebp
   18c99:	f2 41 0f 59 c8       	mulsd  %xmm8,%xmm1
   18c9e:	41 0f 45 ed          	cmovne %r13d,%ebp
   18ca2:	f2 0f 59 c3          	mulsd  %xmm3,%xmm0
   18ca6:	39 ee                	cmp    %ebp,%esi
   18ca8:	0f 4e ee             	cmovle %esi,%ebp
   18cab:	f2 41 0f 5c c9       	subsd  %xmm9,%xmm1
   18cb0:	48 63 ed             	movslq %ebp,%rbp
   18cb3:	4a 8d ac 15 d6 02 00 	lea    0x2d6(%rbp,%r10,1),%rbp
   18cba:	00 
   18cbb:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
   18cbf:	f2 0f 58 ca          	addsd  %xmm2,%xmm1
   18cc3:	f2 0f 58 8c ec c0 0a 	addsd  0xac0(%rsp,%rbp,8),%xmm1
   18cca:	00 00 
   18ccc:	f2 41 0f 5f cc       	maxsd  %xmm12,%xmm1
   18cd1:	83 c1 01             	add    $0x1,%ecx
   18cd4:	66 0f ef c0          	pxor   %xmm0,%xmm0
   18cd8:	45 84 e4             	test   %r12b,%r12b
   18cdb:	74 0a                	je     18ce7 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x357>
   18cdd:	66 0f ef c0          	pxor   %xmm0,%xmm0
   18ce1:	31 db                	xor    %ebx,%ebx
   18ce3:	f2 0f 2a c1          	cvtsi2sd %ecx,%xmm0
   18ce7:	f2 0f 59 c3          	mulsd  %xmm3,%xmm0
   18ceb:	39 de                	cmp    %ebx,%esi
   18ced:	66 44 0f 28 f9       	movapd %xmm1,%xmm15
   18cf2:	0f 4e de             	cmovle %esi,%ebx
   18cf5:	f2 44 0f 58 fc       	addsd  %xmm4,%xmm15
   18cfa:	48 63 db             	movslq %ebx,%rbx
   18cfd:	4a 8d 9c 13 d0 02 00 	lea    0x2d0(%rbx,%r10,1),%rbx
   18d04:	00 
   18d05:	f2 0f 58 c7          	addsd  %xmm7,%xmm0
   18d09:	f2 0f 58 c2          	addsd  %xmm2,%xmm0
   18d0d:	f2 0f 58 84 dc c0 0a 	addsd  0xac0(%rsp,%rbx,8),%xmm0
   18d14:	00 00 
   18d16:	bb 01 00 00 00       	mov    $0x1,%ebx
   18d1b:	66 41 0f 2f c7       	comisd %xmm15,%xmm0
   18d20:	77 06                	ja     18d28 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x398>
   18d22:	66 0f 28 c1          	movapd %xmm1,%xmm0
   18d26:	31 db                	xor    %ebx,%ebx
   18d28:	39 d6                	cmp    %edx,%esi
   18d2a:	7e 5c                	jle    18d88 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x3f8>
   18d2c:	45 84 e4             	test   %r12b,%r12b
   18d2f:	75 5c                	jne    18d8d <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x3fd>
   18d31:	66 0f ef c9          	pxor   %xmm1,%xmm1
   18d35:	f2 0f 59 cb          	mulsd  %xmm3,%xmm1
   18d39:	66 44 0f 28 fe       	movapd %xmm6,%xmm15
   18d3e:	39 ce                	cmp    %ecx,%esi
   18d40:	f2 44 0f 5c fd       	subsd  %xmm5,%xmm15
   18d45:	0f 4e ce             	cmovle %esi,%ecx
   18d48:	48 63 c9             	movslq %ecx,%rcx
   18d4b:	4a 8d 8c 11 d0 02 00 	lea    0x2d0(%rcx,%r10,1),%rcx
   18d52:	00 
   18d53:	f2 41 0f 58 cf       	addsd  %xmm15,%xmm1
   18d58:	66 44 0f 28 f8       	movapd %xmm0,%xmm15
   18d5d:	f2 44 0f 58 fc       	addsd  %xmm4,%xmm15
   18d62:	f2 0f 58 ca          	addsd  %xmm2,%xmm1
   18d66:	f2 0f 58 8c cc c0 0a 	addsd  0xac0(%rsp,%rcx,8),%xmm1
   18d6d:	00 00 
   18d6f:	66 41 0f 2f cf       	comisd %xmm15,%xmm1
   18d74:	0f 87 d6 fe ff ff    	ja     18c50 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x2c0>
   18d7a:	66 0f 28 c8          	movapd %xmm0,%xmm1
   18d7e:	31 c9                	xor    %ecx,%ecx
   18d80:	e9 d5 fe ff ff       	jmp    18c5a <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x2ca>
   18d85:	0f 1f 00             	nopl   (%rax)
   18d88:	45 84 e4             	test   %r12b,%r12b
   18d8b:	74 ed                	je     18d7a <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x3ea>
   18d8d:	66 0f ef c9          	pxor   %xmm1,%xmm1
   18d91:	f2 0f 2a c9          	cvtsi2sd %ecx,%xmm1
   18d95:	b9 01 00 00 00       	mov    $0x1,%ecx
   18d9a:	eb 99                	jmp    18d35 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x3a5>
   18d9c:	0f 1f 40 00          	nopl   0x0(%rax)
   18da0:	48 8b 2c 24          	mov    (%rsp),%rbp
   18da4:	48 8b 5c 24 08       	mov    0x8(%rsp),%rbx
   18da9:	41 83 fd 01          	cmp    $0x1,%r13d
   18dad:	74 19                	je     18dc8 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x438>
   18daf:	85 f6                	test   %esi,%esi
   18db1:	78 15                	js     18dc8 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x438>
   18db3:	49 83 c6 30          	add    $0x30,%r14
   18db7:	41 bd 01 00 00 00    	mov    $0x1,%r13d
   18dbd:	e9 f3 fd ff ff       	jmp    18bb5 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x225>
   18dc2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
   18dc8:	44 8b 44 24 38       	mov    0x38(%rsp),%r8d
   18dcd:	49 89 c3             	mov    %rax,%r11
   18dd0:	83 ef 01             	sub    $0x1,%edi
   18dd3:	49 83 eb 60          	sub    $0x60,%r11
   18dd7:	48 83 ed 48          	sub    $0x48,%rbp
   18ddb:	48 83 eb 01          	sub    $0x1,%rbx
   18ddf:	48 39 5c 24 28       	cmp    %rbx,0x28(%rsp)
   18de4:	0f 85 76 fd ff ff    	jne    18b60 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x1d0>
   18dea:	45 89 c4             	mov    %r8d,%r12d
   18ded:	4d 63 c7             	movslq %r15d,%r8
   18df0:	44 89 f8             	mov    %r15d,%eax
   18df3:	4c 8b 6c 24 50       	mov    0x50(%rsp),%r13
   18df8:	c1 f8 1f             	sar    $0x1f,%eax
   18dfb:	44 8b 5c 24 5c       	mov    0x5c(%rsp),%r11d
   18e00:	66 0f ef db          	pxor   %xmm3,%xmm3
   18e04:	89 34 24             	mov    %esi,(%rsp)
   18e07:	4d 69 c0 67 66 66 66 	imul   $0x66666667,%r8,%r8
   18e0e:	f2 41 0f 10 65 30    	movsd  0x30(%r13),%xmm4
   18e14:	44 8b 6c 24 3c       	mov    0x3c(%rsp),%r13d
   18e19:	4c 8b 4c 24 40       	mov    0x40(%rsp),%r9
   18e1e:	4c 8b 74 24 48       	mov    0x48(%rsp),%r14
   18e23:	44 8b 54 24 58       	mov    0x58(%rsp),%r10d
   18e28:	f2 0f 10 15 68 56 0b 	movsd  0xb5668(%rip),%xmm2        # ce498 <C.1461.14+0xe8>
   18e2f:	00 
   18e30:	49 c1 f8 22          	sar    $0x22,%r8
   18e34:	4c 89 74 24 08       	mov    %r14,0x8(%rsp)
   18e39:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
   18e3e:	41 29 c0             	sub    %eax,%r8d
   18e41:	43 8d 04 80          	lea    (%r8,%r8,4),%eax
   18e45:	44 89 c2             	mov    %r8d,%edx
   18e48:	01 c0                	add    %eax,%eax
   18e4a:	8d 4a fc             	lea    -0x4(%rdx),%ecx
   18e4d:	41 29 c7             	sub    %eax,%r15d
   18e50:	41 8d 47 fc          	lea    -0x4(%r15),%eax
   18e54:	45 89 f8             	mov    %r15d,%r8d
   18e57:	45 8d 7d 01          	lea    0x1(%r13),%r15d
   18e5b:	89 c7                	mov    %eax,%edi
   18e5d:	f7 df                	neg    %edi
   18e5f:	0f 48 f8             	cmovs  %eax,%edi
   18e62:	89 c8                	mov    %ecx,%eax
   18e64:	f7 d8                	neg    %eax
   18e66:	0f 48 c1             	cmovs  %ecx,%eax
   18e69:	41 83 e8 05          	sub    $0x5,%r8d
   18e6d:	44 89 c1             	mov    %r8d,%ecx
   18e70:	f7 d9                	neg    %ecx
   18e72:	41 0f 48 c8          	cmovs  %r8d,%ecx
   18e76:	83 ea 05             	sub    $0x5,%edx
   18e79:	41 89 d0             	mov    %edx,%r8d
   18e7c:	41 f7 d8             	neg    %r8d
   18e7f:	41 0f 49 d0          	cmovns %r8d,%edx
   18e83:	44 8d 04 01          	lea    (%rcx,%rax,1),%r8d
   18e87:	01 f8                	add    %edi,%eax
   18e89:	41 39 c0             	cmp    %eax,%r8d
   18e8c:	41 0f 4e c0          	cmovle %r8d,%eax
   18e90:	41 b8 64 00 00 00    	mov    $0x64,%r8d
   18e96:	44 39 c0             	cmp    %r8d,%eax
   18e99:	41 0f 4f c0          	cmovg  %r8d,%eax
   18e9d:	01 d7                	add    %edx,%edi
   18e9f:	39 f8                	cmp    %edi,%eax
   18ea1:	0f 4f c7             	cmovg  %edi,%eax
   18ea4:	01 ca                	add    %ecx,%edx
   18ea6:	4a 8d 0c cd 00 00 00 	lea    0x0(,%r9,8),%rcx
   18ead:	00 
   18eae:	39 d0                	cmp    %edx,%eax
   18eb0:	0f 4f c2             	cmovg  %edx,%eax
   18eb3:	49 01 c9             	add    %rcx,%r9
   18eb6:	31 db                	xor    %ebx,%ebx
   18eb8:	f2 0f 2a d8          	cvtsi2sd %eax,%xmm3
   18ebc:	f2 0f 59 1d 44 52 0b 	mulsd  0xb5244(%rip),%xmm3        # ce108 <C.449.8+0x80>
   18ec3:	00 
   18ec4:	44 89 e8             	mov    %r13d,%eax
   18ec7:	44 29 d8             	sub    %r11d,%eax
   18eca:	45 31 db             	xor    %r11d,%r11d
   18ecd:	8d 78 01             	lea    0x1(%rax),%edi
   18ed0:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
   18ed5:	44 29 e7             	sub    %r12d,%edi
   18ed8:	4e 8d 04 c8          	lea    (%rax,%r9,8),%r8
   18edc:	48 01 c1             	add    %rax,%rcx
   18edf:	e9 3d 01 00 00       	jmp    19021 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x691>
   18ee4:	0f 1f 40 00          	nopl   0x0(%rax)
   18ee8:	49 63 c3             	movslq %r11d,%rax
   18eeb:	4d 63 ca             	movslq %r10d,%r9
   18eee:	48 63 d3             	movslq %ebx,%rdx
   18ef1:	f2 41 0f 10 00       	movsd  (%r8),%xmm0
   18ef6:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
   18efa:	4f 8d 0c 49          	lea    (%r9,%r9,2),%r9
   18efe:	66 0f ef c9          	pxor   %xmm1,%xmm1
   18f02:	44 89 fd             	mov    %r15d,%ebp
   18f05:	48 01 c0             	add    %rax,%rax
   18f08:	4e 8d 0c 88          	lea    (%rax,%r9,4),%r9
   18f0c:	49 01 d1             	add    %rdx,%r9
   18f0f:	31 d2                	xor    %edx,%edx
   18f11:	49 c1 e1 04          	shl    $0x4,%r9
   18f15:	42 8b 84 0c c0 0a 00 	mov    0xac0(%rsp,%r9,1),%eax
   18f1c:	00 
   18f1d:	46 8b 8c 0c c4 0a 00 	mov    0xac4(%rsp,%r9,1),%r9d
   18f24:	00 
   18f25:	85 c0                	test   %eax,%eax
   18f27:	0f 95 c2             	setne  %dl
   18f2a:	45 85 c9             	test   %r9d,%r9d
   18f2d:	f2 0f 2a ca          	cvtsi2sd %edx,%xmm1
   18f31:	41 0f 95 c1          	setne  %r9b
   18f35:	41 0f 95 c6          	setne  %r14b
   18f39:	45 0f b6 c9          	movzbl %r9b,%r9d
   18f3d:	42 8d 54 0a 01       	lea    0x1(%rdx,%r9,1),%edx
   18f42:	f2 0f 5c c1          	subsd  %xmm1,%xmm0
   18f46:	f2 41 0f 11 00       	movsd  %xmm0,(%r8)
   18f4b:	66 0f ef c0          	pxor   %xmm0,%xmm0
   18f4f:	f2 0f 2a c2          	cvtsi2sd %edx,%xmm0
   18f53:	f2 0f 58 c3          	addsd  %xmm3,%xmm0
   18f57:	f2 0f 59 c4          	mulsd  %xmm4,%xmm0
   18f5b:	f2 0f 58 81 70 08 00 	addsd  0x870(%rcx),%xmm0
   18f62:	00 
   18f63:	f2 0f 11 81 70 08 00 	movsd  %xmm0,0x870(%rcx)
   18f6a:	00 
   18f6b:	85 c0                	test   %eax,%eax
   18f6d:	0f 85 43 03 00 00    	jne    192b6 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x926>
   18f73:	41 83 c3 01          	add    $0x1,%r11d
   18f77:	41 83 fb 02          	cmp    $0x2,%r11d
   18f7b:	0f 84 23 03 00 00    	je     192a4 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x914>
   18f81:	42 8d 04 27          	lea    (%rdi,%r12,1),%eax
   18f85:	4d 63 cf             	movslq %r15d,%r9
   18f88:	44 39 e0             	cmp    %r12d,%eax
   18f8b:	0f 8c 7f 01 00 00    	jl     19110 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x780>
   18f91:	89 f8                	mov    %edi,%eax
   18f93:	4c 8d 35 e6 4f 0b 00 	lea    0xb4fe6(%rip),%r14        # cdf80 <_ZN3dp7L9aintervalE>
   18f9a:	99                   	cltd
   18f9b:	41 f7 3c b6          	idivl  (%r14,%rsi,4)
   18f9f:	85 d2                	test   %edx,%edx
   18fa1:	0f 85 69 01 00 00    	jne    19110 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x780>
   18fa7:	48 8d 05 b2 4f 0b 00 	lea    0xb4fb2(%rip),%rax        # cdf60 <_ZN3dp7L7productE>
   18fae:	66 0f 28 c2          	movapd %xmm2,%xmm0
   18fb2:	48 63 14 b0          	movslq (%rax,%rsi,4),%rdx
   18fb6:	4a 8d 04 cd 00 00 00 	lea    0x0(,%r9,8),%rax
   18fbd:	00 
   18fbe:	4a 8d 1c 08          	lea    (%rax,%r9,1),%rbx
   18fc2:	48 01 da             	add    %rbx,%rdx
   18fc5:	31 db                	xor    %ebx,%ebx
   18fc7:	f2 0f 10 4c d4 60    	movsd  0x60(%rsp,%rdx,8),%xmm1
   18fcd:	f2 0f 58 ca          	addsd  %xmm2,%xmm1
   18fd1:	f2 0f 11 4c d4 60    	movsd  %xmm1,0x60(%rsp,%rdx,8)
   18fd7:	f2 42 0f 10 8c cc d0 	movsd  0x8d0(%rsp,%r9,8),%xmm1
   18fde:	08 00 00 
   18fe1:	f2 0f 58 ca          	addsd  %xmm2,%xmm1
   18fe5:	f2 42 0f 11 8c cc d0 	movsd  %xmm1,0x8d0(%rsp,%r9,8)
   18fec:	08 00 00 
   18fef:	90                   	nop
   18ff0:	4c 01 c8             	add    %r9,%rax
   18ff3:	41 83 c2 01          	add    $0x1,%r10d
   18ff7:	41 83 c7 01          	add    $0x1,%r15d
   18ffb:	83 c7 01             	add    $0x1,%edi
   18ffe:	f2 0f 58 84 c4 a0 00 	addsd  0xa0(%rsp,%rax,8),%xmm0
   19005:	00 00 
   19007:	49 83 c0 48          	add    $0x48,%r8
   1900b:	48 83 c1 08          	add    $0x8,%rcx
   1900f:	f2 0f 11 84 c4 a0 00 	movsd  %xmm0,0xa0(%rsp,%rax,8)
   19016:	00 00 
   19018:	83 fd 1d             	cmp    $0x1d,%ebp
   1901b:	0f 84 00 01 00 00    	je     19121 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x791>
   19021:	45 39 d5             	cmp    %r10d,%r13d
   19024:	0f 85 be fe ff ff    	jne    18ee8 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x558>
   1902a:	f2 41 0f 10 08       	movsd  (%r8),%xmm1
   1902f:	42 8d 04 27          	lea    (%rdi,%r12,1),%eax
   19033:	66 0f 28 c2          	movapd %xmm2,%xmm0
   19037:	44 89 fd             	mov    %r15d,%ebp
   1903a:	4d 63 cf             	movslq %r15d,%r9
   1903d:	f2 0f 5c ca          	subsd  %xmm2,%xmm1
   19041:	f2 41 0f 11 08       	movsd  %xmm1,(%r8)
   19046:	66 0f 28 cb          	movapd %xmm3,%xmm1
   1904a:	f2 41 0f 58 ce       	addsd  %xmm14,%xmm1
   1904f:	f2 0f 59 cc          	mulsd  %xmm4,%xmm1
   19053:	f2 0f 58 89 70 08 00 	addsd  0x870(%rcx),%xmm1
   1905a:	00 
   1905b:	f2 0f 11 89 70 08 00 	movsd  %xmm1,0x870(%rcx)
   19062:	00 
   19063:	41 39 c4             	cmp    %eax,%r12d
   19066:	0f 8f 84 00 00 00    	jg     190f0 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x760>
   1906c:	89 f8                	mov    %edi,%eax
   1906e:	4c 8d 1d 0b 4f 0b 00 	lea    0xb4f0b(%rip),%r11        # cdf80 <_ZN3dp7L9aintervalE>
   19075:	41 be 01 00 00 00    	mov    $0x1,%r14d
   1907b:	99                   	cltd
   1907c:	41 f7 3c b3          	idivl  (%r11,%rsi,4)
   19080:	85 d2                	test   %edx,%edx
   19082:	75 6c                	jne    190f0 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x760>
   19084:	48 8d 05 d5 4e 0b 00 	lea    0xb4ed5(%rip),%rax        # cdf60 <_ZN3dp7L7productE>
   1908b:	83 c3 01             	add    $0x1,%ebx
   1908e:	66 0f ef c9          	pxor   %xmm1,%xmm1
   19092:	48 63 14 b0          	movslq (%rax,%rsi,4),%rdx
   19096:	f2 0f 2a cb          	cvtsi2sd %ebx,%xmm1
   1909a:	4a 8d 04 cd 00 00 00 	lea    0x0(,%r9,8),%rax
   190a1:	00 
   190a2:	31 db                	xor    %ebx,%ebx
   190a4:	4e 8d 1c 08          	lea    (%rax,%r9,1),%r11
   190a8:	4c 01 da             	add    %r11,%rdx
   190ab:	f2 0f 58 4c d4 60    	addsd  0x60(%rsp,%rdx,8),%xmm1
   190b1:	f2 0f 11 4c d4 60    	movsd  %xmm1,0x60(%rsp,%rdx,8)
   190b7:	f2 42 0f 10 8c cc d0 	movsd  0x8d0(%rsp,%r9,8),%xmm1
   190be:	08 00 00 
   190c1:	f2 0f 58 c8          	addsd  %xmm0,%xmm1
   190c5:	f2 42 0f 11 8c cc d0 	movsd  %xmm1,0x8d0(%rsp,%r9,8)
   190cc:	08 00 00 
   190cf:	45 31 db             	xor    %r11d,%r11d
   190d2:	45 84 f6             	test   %r14b,%r14b
   190d5:	0f 84 15 ff ff ff    	je     18ff0 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x660>
   190db:	8b 14 24             	mov    (%rsp),%edx
   190de:	83 c3 01             	add    $0x1,%ebx
   190e1:	39 d3                	cmp    %edx,%ebx
   190e3:	0f 4f da             	cmovg  %edx,%ebx
   190e6:	45 31 db             	xor    %r11d,%r11d
   190e9:	e9 02 ff ff ff       	jmp    18ff0 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x660>
   190ee:	66 90                	xchg   %ax,%ax
   190f0:	8b 14 24             	mov    (%rsp),%edx
   190f3:	83 c3 01             	add    $0x1,%ebx
   190f6:	4a 8d 04 cd 00 00 00 	lea    0x0(,%r9,8),%rax
   190fd:	00 
   190fe:	39 d3                	cmp    %edx,%ebx
   19100:	0f 4f da             	cmovg  %edx,%ebx
   19103:	45 31 db             	xor    %r11d,%r11d
   19106:	e9 e5 fe ff ff       	jmp    18ff0 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x660>
   1910b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
   19110:	66 0f 28 c2          	movapd %xmm2,%xmm0
   19114:	4a 8d 04 cd 00 00 00 	lea    0x0(,%r9,8),%rax
   1911b:	00 
   1911c:	e9 cf fe ff ff       	jmp    18ff0 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x660>
   19121:	4c 8b 74 24 08       	mov    0x8(%rsp),%r14
   19126:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
   1912b:	b9 4c 01 00 00       	mov    $0x14c,%ecx
   19130:	4c 89 f7             	mov    %r14,%rdi
   19133:	f3 48 a5             	rep movsq %ds:(%rsi),%es:(%rdi)
   19136:	48 81 c4 88 2c 00 00 	add    $0x2c88,%rsp
   1913d:	4c 89 f0             	mov    %r14,%rax
   19140:	5b                   	pop    %rbx
   19141:	5d                   	pop    %rbp
   19142:	41 5c                	pop    %r12
   19144:	41 5d                	pop    %r13
   19146:	41 5e                	pop    %r14
   19148:	41 5f                	pop    %r15
   1914a:	c3                   	ret
   1914b:	48 8d b6 00 01 00 00 	lea    0x100(%rsi),%rsi
   19152:	45 31 c9             	xor    %r9d,%r9d
   19155:	e8 66 e8 01 00       	call   379c0 <_ZNK11competitive7Planner13animal_streamEiiiPKN7fastkag4TileE>
   1915a:	f2 41 0f 10 45 30    	movsd  0x30(%r13),%xmm0
   19160:	66 41 0f 10 8e 70 08 	movupd 0x870(%r14),%xmm1
   19167:	00 00 
   19169:	66 41 0f 10 ae 50 09 	movupd 0x950(%r14),%xmm5
   19170:	00 00 
   19172:	66 0f 14 c0          	unpcklpd %xmm0,%xmm0
   19176:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1917a:	41 0f 11 8e 70 08 00 	movups %xmm1,0x870(%r14)
   19181:	00 
   19182:	66 41 0f 10 8e 80 08 	movupd 0x880(%r14),%xmm1
   19189:	00 00 
   1918b:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1918f:	41 0f 11 8e 80 08 00 	movups %xmm1,0x880(%r14)
   19196:	00 
   19197:	66 41 0f 10 8e 90 08 	movupd 0x890(%r14),%xmm1
   1919e:	00 00 
   191a0:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   191a4:	41 0f 11 8e 90 08 00 	movups %xmm1,0x890(%r14)
   191ab:	00 
   191ac:	66 41 0f 10 8e a0 08 	movupd 0x8a0(%r14),%xmm1
   191b3:	00 00 
   191b5:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   191b9:	41 0f 11 8e a0 08 00 	movups %xmm1,0x8a0(%r14)
   191c0:	00 
   191c1:	66 41 0f 10 8e b0 08 	movupd 0x8b0(%r14),%xmm1
   191c8:	00 00 
   191ca:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   191ce:	41 0f 11 8e b0 08 00 	movups %xmm1,0x8b0(%r14)
   191d5:	00 
   191d6:	66 41 0f 10 8e c0 08 	movupd 0x8c0(%r14),%xmm1
   191dd:	00 00 
   191df:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   191e3:	41 0f 11 8e c0 08 00 	movups %xmm1,0x8c0(%r14)
   191ea:	00 
   191eb:	66 41 0f 10 8e d0 08 	movupd 0x8d0(%r14),%xmm1
   191f2:	00 00 
   191f4:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   191f8:	41 0f 11 8e d0 08 00 	movups %xmm1,0x8d0(%r14)
   191ff:	00 
   19200:	66 41 0f 10 8e e0 08 	movupd 0x8e0(%r14),%xmm1
   19207:	00 00 
   19209:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1920d:	41 0f 11 8e e0 08 00 	movups %xmm1,0x8e0(%r14)
   19214:	00 
   19215:	66 41 0f 10 8e f0 08 	movupd 0x8f0(%r14),%xmm1
   1921c:	00 00 
   1921e:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   19222:	41 0f 11 8e f0 08 00 	movups %xmm1,0x8f0(%r14)
   19229:	00 
   1922a:	66 41 0f 10 8e 00 09 	movupd 0x900(%r14),%xmm1
   19231:	00 00 
   19233:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   19237:	41 0f 11 8e 00 09 00 	movups %xmm1,0x900(%r14)
   1923e:	00 
   1923f:	66 41 0f 10 8e 10 09 	movupd 0x910(%r14),%xmm1
   19246:	00 00 
   19248:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1924c:	41 0f 11 8e 10 09 00 	movups %xmm1,0x910(%r14)
   19253:	00 
   19254:	66 41 0f 10 8e 20 09 	movupd 0x920(%r14),%xmm1
   1925b:	00 00 
   1925d:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   19261:	41 0f 11 8e 20 09 00 	movups %xmm1,0x920(%r14)
   19268:	00 
   19269:	66 41 0f 10 8e 30 09 	movupd 0x930(%r14),%xmm1
   19270:	00 00 
   19272:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   19276:	41 0f 11 8e 30 09 00 	movups %xmm1,0x930(%r14)
   1927d:	00 
   1927e:	66 41 0f 10 8e 40 09 	movupd 0x940(%r14),%xmm1
   19285:	00 00 
   19287:	66 0f 59 c8          	mulpd  %xmm0,%xmm1
   1928b:	66 0f 59 c5          	mulpd  %xmm5,%xmm0
   1928f:	41 0f 11 8e 40 09 00 	movups %xmm1,0x940(%r14)
   19296:	00 
   19297:	41 0f 11 86 50 09 00 	movups %xmm0,0x950(%r14)
   1929e:	00 
   1929f:	e9 92 fe ff ff       	jmp    19136 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x7a6>
   192a4:	44 89 bc 24 b0 0a 00 	mov    %r15d,0xab0(%rsp)
   192ab:	00 
   192ac:	4c 8b 74 24 08       	mov    0x8(%rsp),%r14
   192b1:	e9 70 fe ff ff       	jmp    19126 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x796>
   192b6:	42 8d 04 27          	lea    (%rdi,%r12,1),%eax
   192ba:	4d 63 cf             	movslq %r15d,%r9
   192bd:	41 39 c4             	cmp    %eax,%r12d
   192c0:	7e 15                	jle    192d7 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x947>
   192c2:	f2 0f 10 05 ce 51 0b 	movsd  0xb51ce(%rip),%xmm0        # ce498 <C.1461.14+0xe8>
   192c9:	00 
   192ca:	4a 8d 04 cd 00 00 00 	lea    0x0(,%r9,8),%rax
   192d1:	00 
   192d2:	e9 f8 fd ff ff       	jmp    190cf <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x73f>
   192d7:	89 f8                	mov    %edi,%eax
   192d9:	4c 8d 1d a0 4c 0b 00 	lea    0xb4ca0(%rip),%r11        # cdf80 <_ZN3dp7L9aintervalE>
   192e0:	f2 0f 10 05 b0 51 0b 	movsd  0xb51b0(%rip),%xmm0        # ce498 <C.1461.14+0xe8>
   192e7:	00 
   192e8:	99                   	cltd
   192e9:	41 f7 3c b3          	idivl  (%r11,%rsi,4)
   192ed:	85 d2                	test   %edx,%edx
   192ef:	0f 84 8f fd ff ff    	je     19084 <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x6f4>
   192f5:	4a 8d 04 cd 00 00 00 	lea    0x0(,%r9,8),%rax
   192fc:	00 
   192fd:	e9 cd fd ff ff       	jmp    190cf <_ZNK5triad10Controller11animal_pathEiiiRKSt5arrayIS1_IdLm9EELm30EEPKN7fastkag4TileE.constprop.0+0x73f>

Disassembly of section .fini:
