��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.0.02v2.0.0-rc2-26-g64c3d388��
�
sequential/conv2d/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape: *)
shared_namesequential/conv2d/kernel
�
,sequential/conv2d/kernel/Read/ReadVariableOpReadVariableOpsequential/conv2d/kernel*
dtype0*&
_output_shapes
: 
�
sequential/conv2d/biasVarHandleOp*
shape: *'
shared_namesequential/conv2d/bias*
dtype0*
_output_shapes
: 
}
*sequential/conv2d/bias/Read/ReadVariableOpReadVariableOpsequential/conv2d/bias*
_output_shapes
: *
dtype0
�
sequential/p_re_lu/alphaVarHandleOp*
dtype0*
_output_shapes
: *
shape::� *)
shared_namesequential/p_re_lu/alpha
�
,sequential/p_re_lu/alpha/Read/ReadVariableOpReadVariableOpsequential/p_re_lu/alpha*
dtype0*#
_output_shapes
::� 
�
sequential/conv2d_1/kernelVarHandleOp*+
shared_namesequential/conv2d_1/kernel*
dtype0*
_output_shapes
: *
shape: @
�
.sequential/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpsequential/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
�
sequential/conv2d_1/biasVarHandleOp*)
shared_namesequential/conv2d_1/bias*
dtype0*
_output_shapes
: *
shape:@
�
,sequential/conv2d_1/bias/Read/ReadVariableOpReadVariableOpsequential/conv2d_1/bias*
dtype0*
_output_shapes
:@
�
sequential/p_re_lu_1/alphaVarHandleOp*+
shared_namesequential/p_re_lu_1/alpha*
dtype0*
_output_shapes
: *
shape:K@
�
.sequential/p_re_lu_1/alpha/Read/ReadVariableOpReadVariableOpsequential/p_re_lu_1/alpha*
dtype0*"
_output_shapes
:K@
�
sequential/conv2d_2/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:@�*+
shared_namesequential/conv2d_2/kernel
�
.sequential/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpsequential/conv2d_2/kernel*
dtype0*'
_output_shapes
:@�
�
sequential/conv2d_2/biasVarHandleOp*)
shared_namesequential/conv2d_2/bias*
dtype0*
_output_shapes
: *
shape:�
�
,sequential/conv2d_2/bias/Read/ReadVariableOpReadVariableOpsequential/conv2d_2/bias*
dtype0*
_output_shapes	
:�
�
sequential/p_re_lu_2/alphaVarHandleOp*
shape:!�*+
shared_namesequential/p_re_lu_2/alpha*
dtype0*
_output_shapes
: 
�
.sequential/p_re_lu_2/alpha/Read/ReadVariableOpReadVariableOpsequential/p_re_lu_2/alpha*
dtype0*#
_output_shapes
:!�
�
sequential/dense/kernelVarHandleOp*(
shared_namesequential/dense/kernel*
dtype0*
_output_shapes
: *
shape:
�@�
�
+sequential/dense/kernel/Read/ReadVariableOpReadVariableOpsequential/dense/kernel* 
_output_shapes
:
�@�*
dtype0
�
sequential/dense/biasVarHandleOp*
shape:�*&
shared_namesequential/dense/bias*
dtype0*
_output_shapes
: 
|
)sequential/dense/bias/Read/ReadVariableOpReadVariableOpsequential/dense/bias*
dtype0*
_output_shapes	
:�
f
	Adam/iterVarHandleOp*
dtype0	*
_output_shapes
: *
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
shared_name
Adam/decay*
dtype0*
_output_shapes
: *
shape: 
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
shape: *#
shared_nameAdam/learning_rate*
dtype0
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shared_nametotal*
dtype0*
_output_shapes
: *
shape: 
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
�
Adam/sequential/conv2d/kernel/mVarHandleOp*0
shared_name!Adam/sequential/conv2d/kernel/m*
dtype0*
_output_shapes
: *
shape: 
�
3Adam/sequential/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d/kernel/m*
dtype0*&
_output_shapes
: 
�
Adam/sequential/conv2d/bias/mVarHandleOp*
shape: *.
shared_nameAdam/sequential/conv2d/bias/m*
dtype0*
_output_shapes
: 
�
1Adam/sequential/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d/bias/m*
dtype0*
_output_shapes
: 
�
Adam/sequential/p_re_lu/alpha/mVarHandleOp*
dtype0*
_output_shapes
: *
shape::� *0
shared_name!Adam/sequential/p_re_lu/alpha/m
�
3Adam/sequential/p_re_lu/alpha/m/Read/ReadVariableOpReadVariableOpAdam/sequential/p_re_lu/alpha/m*
dtype0*#
_output_shapes
::� 
�
!Adam/sequential/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
shape: @*2
shared_name#!Adam/sequential/conv2d_1/kernel/m*
dtype0
�
5Adam/sequential/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/sequential/conv2d_1/kernel/m*
dtype0*&
_output_shapes
: @
�
Adam/sequential/conv2d_1/bias/mVarHandleOp*0
shared_name!Adam/sequential/conv2d_1/bias/m*
dtype0*
_output_shapes
: *
shape:@
�
3Adam/sequential/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d_1/bias/m*
dtype0*
_output_shapes
:@
�
!Adam/sequential/p_re_lu_1/alpha/mVarHandleOp*
_output_shapes
: *
shape:K@*2
shared_name#!Adam/sequential/p_re_lu_1/alpha/m*
dtype0
�
5Adam/sequential/p_re_lu_1/alpha/m/Read/ReadVariableOpReadVariableOp!Adam/sequential/p_re_lu_1/alpha/m*
dtype0*"
_output_shapes
:K@
�
!Adam/sequential/conv2d_2/kernel/mVarHandleOp*
shape:@�*2
shared_name#!Adam/sequential/conv2d_2/kernel/m*
dtype0*
_output_shapes
: 
�
5Adam/sequential/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/sequential/conv2d_2/kernel/m*
dtype0*'
_output_shapes
:@�
�
Adam/sequential/conv2d_2/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*0
shared_name!Adam/sequential/conv2d_2/bias/m
�
3Adam/sequential/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d_2/bias/m*
dtype0*
_output_shapes	
:�
�
!Adam/sequential/p_re_lu_2/alpha/mVarHandleOp*
_output_shapes
: *
shape:!�*2
shared_name#!Adam/sequential/p_re_lu_2/alpha/m*
dtype0
�
5Adam/sequential/p_re_lu_2/alpha/m/Read/ReadVariableOpReadVariableOp!Adam/sequential/p_re_lu_2/alpha/m*
dtype0*#
_output_shapes
:!�
�
Adam/sequential/dense/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:
�@�*/
shared_name Adam/sequential/dense/kernel/m
�
2Adam/sequential/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/kernel/m*
dtype0* 
_output_shapes
:
�@�
�
Adam/sequential/dense/bias/mVarHandleOp*
shape:�*-
shared_nameAdam/sequential/dense/bias/m*
dtype0*
_output_shapes
: 
�
0Adam/sequential/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/bias/m*
dtype0*
_output_shapes	
:�
�
Adam/sequential/conv2d/kernel/vVarHandleOp*0
shared_name!Adam/sequential/conv2d/kernel/v*
dtype0*
_output_shapes
: *
shape: 
�
3Adam/sequential/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d/kernel/v*
dtype0*&
_output_shapes
: 
�
Adam/sequential/conv2d/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: *.
shared_nameAdam/sequential/conv2d/bias/v
�
1Adam/sequential/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d/bias/v*
dtype0*
_output_shapes
: 
�
Adam/sequential/p_re_lu/alpha/vVarHandleOp*0
shared_name!Adam/sequential/p_re_lu/alpha/v*
dtype0*
_output_shapes
: *
shape::� 
�
3Adam/sequential/p_re_lu/alpha/v/Read/ReadVariableOpReadVariableOpAdam/sequential/p_re_lu/alpha/v*#
_output_shapes
::� *
dtype0
�
!Adam/sequential/conv2d_1/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: @*2
shared_name#!Adam/sequential/conv2d_1/kernel/v
�
5Adam/sequential/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/sequential/conv2d_1/kernel/v*
dtype0*&
_output_shapes
: @
�
Adam/sequential/conv2d_1/bias/vVarHandleOp*0
shared_name!Adam/sequential/conv2d_1/bias/v*
dtype0*
_output_shapes
: *
shape:@
�
3Adam/sequential/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d_1/bias/v*
dtype0*
_output_shapes
:@
�
!Adam/sequential/p_re_lu_1/alpha/vVarHandleOp*2
shared_name#!Adam/sequential/p_re_lu_1/alpha/v*
dtype0*
_output_shapes
: *
shape:K@
�
5Adam/sequential/p_re_lu_1/alpha/v/Read/ReadVariableOpReadVariableOp!Adam/sequential/p_re_lu_1/alpha/v*
dtype0*"
_output_shapes
:K@
�
!Adam/sequential/conv2d_2/kernel/vVarHandleOp*
shape:@�*2
shared_name#!Adam/sequential/conv2d_2/kernel/v*
dtype0*
_output_shapes
: 
�
5Adam/sequential/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/sequential/conv2d_2/kernel/v*
dtype0*'
_output_shapes
:@�
�
Adam/sequential/conv2d_2/bias/vVarHandleOp*0
shared_name!Adam/sequential/conv2d_2/bias/v*
dtype0*
_output_shapes
: *
shape:�
�
3Adam/sequential/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d_2/bias/v*
dtype0*
_output_shapes	
:�
�
!Adam/sequential/p_re_lu_2/alpha/vVarHandleOp*2
shared_name#!Adam/sequential/p_re_lu_2/alpha/v*
dtype0*
_output_shapes
: *
shape:!�
�
5Adam/sequential/p_re_lu_2/alpha/v/Read/ReadVariableOpReadVariableOp!Adam/sequential/p_re_lu_2/alpha/v*
dtype0*#
_output_shapes
:!�
�
Adam/sequential/dense/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:
�@�*/
shared_name Adam/sequential/dense/kernel/v
�
2Adam/sequential/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/kernel/v*
dtype0* 
_output_shapes
:
�@�
�
Adam/sequential/dense/bias/vVarHandleOp*-
shared_nameAdam/sequential/dense/bias/v*
dtype0*
_output_shapes
: *
shape:�
�
0Adam/sequential/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/bias/v*
_output_shapes	
:�*
dtype0

NoOpNoOp
�I
ConstConst"/device:CPU:0*�I
value�IB�I B�I
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
	optimizer

signatures
layer_with_weights-0
layer_with_weights-1
layer_with_weights-2
layer_with_weights-3
layer_with_weights-4
layer_with_weights-5
layer_with_weights-6
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
]
	alpha
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
 regularization_losses
!	variables
"	keras_api
h

#kernel
$bias
%trainable_variables
&regularization_losses
'	variables
(	keras_api
]
	)alpha
*trainable_variables
+regularization_losses
,	variables
-	keras_api
R
.trainable_variables
/regularization_losses
0	variables
1	keras_api
h

2kernel
3bias
4trainable_variables
5regularization_losses
6	variables
7	keras_api
]
	8alpha
9trainable_variables
:regularization_losses
;	variables
<	keras_api
R
=trainable_variables
>regularization_losses
?	variables
@	keras_api
R
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
h

Ekernel
Fbias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
R
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
R
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
�
Siter

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_ratem�m�m�#m�$m�)m�2m�3m�8m�Em�Fm�v�v�v�#v�$v�)v�2v�3v�8v�Ev�Fv�
 
N
0
1
2
#3
$4
)5
26
37
88
E9
F10
 
N
0
1
2
#3
$4
)5
26
37
88
E9
F10
�
Xmetrics
Ylayer_regularization_losses
Znon_trainable_variables
trainable_variables

[layers
regularization_losses
	variables
WU
VARIABLE_VALUEsequential/conv2d/kernel)layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEsequential/conv2d/bias'layer-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
\metrics
]layer_regularization_losses
^non_trainable_variables

_layers
trainable_variables
regularization_losses
	variables
VT
VARIABLE_VALUEsequential/p_re_lu/alpha(layer-1/alpha/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
�
`metrics
alayer_regularization_losses
bnon_trainable_variables

clayers
trainable_variables
regularization_losses
	variables
 
 
 
�
dmetrics
elayer_regularization_losses
fnon_trainable_variables

glayers
trainable_variables
 regularization_losses
!	variables
YW
VARIABLE_VALUEsequential/conv2d_1/kernel)layer-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEsequential/conv2d_1/bias'layer-3/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1
�
hmetrics
ilayer_regularization_losses
jnon_trainable_variables

klayers
%trainable_variables
&regularization_losses
'	variables
XV
VARIABLE_VALUEsequential/p_re_lu_1/alpha(layer-4/alpha/.ATTRIBUTES/VARIABLE_VALUE

)0
 

)0
�
lmetrics
mlayer_regularization_losses
nnon_trainable_variables

olayers
*trainable_variables
+regularization_losses
,	variables
 
 
 
�
pmetrics
qlayer_regularization_losses
rnon_trainable_variables

slayers
.trainable_variables
/regularization_losses
0	variables
YW
VARIABLE_VALUEsequential/conv2d_2/kernel)layer-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEsequential/conv2d_2/bias'layer-6/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31
 

20
31
�
tmetrics
ulayer_regularization_losses
vnon_trainable_variables

wlayers
4trainable_variables
5regularization_losses
6	variables
XV
VARIABLE_VALUEsequential/p_re_lu_2/alpha(layer-7/alpha/.ATTRIBUTES/VARIABLE_VALUE

80
 

80
�
xmetrics
ylayer_regularization_losses
znon_trainable_variables

{layers
9trainable_variables
:regularization_losses
;	variables
 
 
 
�
|metrics
}layer_regularization_losses
~non_trainable_variables

layers
=trainable_variables
>regularization_losses
?	variables
 
 
 
�
�metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
Atrainable_variables
Bregularization_losses
C	variables
WU
VARIABLE_VALUEsequential/dense/kernel*layer-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEsequential/dense/bias(layer-10/bias/.ATTRIBUTES/VARIABLE_VALUE

E0
F1
 

E0
F1
�
�metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
Gtrainable_variables
Hregularization_losses
I	variables
 
 
 
�
�metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
Ktrainable_variables
Lregularization_losses
M	variables
 
 
 
�
�metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
Otrainable_variables
Pregularization_losses
Q	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

�0
 
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


�total

�count
�
_fn_kwargs
�trainable_variables
�regularization_losses
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

�0
�1
�
�metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
�trainable_variables
�regularization_losses
�	variables
 
 

�0
�1
 
zx
VARIABLE_VALUEAdam/sequential/conv2d/kernel/mElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/sequential/conv2d/bias/mClayer-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/sequential/p_re_lu/alpha/mDlayer-1/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE!Adam/sequential/conv2d_1/kernel/mElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/sequential/conv2d_1/bias/mClayer-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE!Adam/sequential/p_re_lu_1/alpha/mDlayer-4/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE!Adam/sequential/conv2d_2/kernel/mElayer-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/sequential/conv2d_2/bias/mClayer-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE!Adam/sequential/p_re_lu_2/alpha/mDlayer-7/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/sequential/dense/kernel/mFlayer-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/sequential/dense/bias/mDlayer-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/sequential/conv2d/kernel/vElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/sequential/conv2d/bias/vClayer-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/sequential/p_re_lu/alpha/vDlayer-1/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE!Adam/sequential/conv2d_1/kernel/vElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/sequential/conv2d_1/bias/vClayer-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE!Adam/sequential/p_re_lu_1/alpha/vDlayer-4/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE!Adam/sequential/conv2d_2/kernel/vElayer-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/sequential/conv2d_2/bias/vClayer-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE!Adam/sequential/p_re_lu_2/alpha/vDlayer-7/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/sequential/dense/kernel/vFlayer-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/sequential/dense/bias/vDlayer-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
�
serving_default_input_1Placeholder*
dtype0*0
_output_shapes
:���������<�*%
shape:���������<�
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1sequential/conv2d/kernelsequential/conv2d/biassequential/p_re_lu/alphasequential/conv2d_1/kernelsequential/conv2d_1/biassequential/p_re_lu_1/alphasequential/conv2d_2/kernelsequential/conv2d_2/biassequential/p_re_lu_2/alphasequential/dense/kernelsequential/dense/bias*-
_gradient_op_typePartitionedCall-119548*-
f(R&
$__inference_signature_wrapper_119462*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������>
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,sequential/conv2d/kernel/Read/ReadVariableOp*sequential/conv2d/bias/Read/ReadVariableOp,sequential/p_re_lu/alpha/Read/ReadVariableOp.sequential/conv2d_1/kernel/Read/ReadVariableOp,sequential/conv2d_1/bias/Read/ReadVariableOp.sequential/p_re_lu_1/alpha/Read/ReadVariableOp.sequential/conv2d_2/kernel/Read/ReadVariableOp,sequential/conv2d_2/bias/Read/ReadVariableOp.sequential/p_re_lu_2/alpha/Read/ReadVariableOp+sequential/dense/kernel/Read/ReadVariableOp)sequential/dense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp3Adam/sequential/conv2d/kernel/m/Read/ReadVariableOp1Adam/sequential/conv2d/bias/m/Read/ReadVariableOp3Adam/sequential/p_re_lu/alpha/m/Read/ReadVariableOp5Adam/sequential/conv2d_1/kernel/m/Read/ReadVariableOp3Adam/sequential/conv2d_1/bias/m/Read/ReadVariableOp5Adam/sequential/p_re_lu_1/alpha/m/Read/ReadVariableOp5Adam/sequential/conv2d_2/kernel/m/Read/ReadVariableOp3Adam/sequential/conv2d_2/bias/m/Read/ReadVariableOp5Adam/sequential/p_re_lu_2/alpha/m/Read/ReadVariableOp2Adam/sequential/dense/kernel/m/Read/ReadVariableOp0Adam/sequential/dense/bias/m/Read/ReadVariableOp3Adam/sequential/conv2d/kernel/v/Read/ReadVariableOp1Adam/sequential/conv2d/bias/v/Read/ReadVariableOp3Adam/sequential/p_re_lu/alpha/v/Read/ReadVariableOp5Adam/sequential/conv2d_1/kernel/v/Read/ReadVariableOp3Adam/sequential/conv2d_1/bias/v/Read/ReadVariableOp5Adam/sequential/p_re_lu_1/alpha/v/Read/ReadVariableOp5Adam/sequential/conv2d_2/kernel/v/Read/ReadVariableOp3Adam/sequential/conv2d_2/bias/v/Read/ReadVariableOp5Adam/sequential/p_re_lu_2/alpha/v/Read/ReadVariableOp2Adam/sequential/dense/kernel/v/Read/ReadVariableOp0Adam/sequential/dense/bias/v/Read/ReadVariableOpConst*-
_gradient_op_typePartitionedCall-119610*(
f#R!
__inference__traced_save_119609*
Tout
2**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *5
Tin.
,2*	
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesequential/conv2d/kernelsequential/conv2d/biassequential/p_re_lu/alphasequential/conv2d_1/kernelsequential/conv2d_1/biassequential/p_re_lu_1/alphasequential/conv2d_2/kernelsequential/conv2d_2/biassequential/p_re_lu_2/alphasequential/dense/kernelsequential/dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/sequential/conv2d/kernel/mAdam/sequential/conv2d/bias/mAdam/sequential/p_re_lu/alpha/m!Adam/sequential/conv2d_1/kernel/mAdam/sequential/conv2d_1/bias/m!Adam/sequential/p_re_lu_1/alpha/m!Adam/sequential/conv2d_2/kernel/mAdam/sequential/conv2d_2/bias/m!Adam/sequential/p_re_lu_2/alpha/mAdam/sequential/dense/kernel/mAdam/sequential/dense/bias/mAdam/sequential/conv2d/kernel/vAdam/sequential/conv2d/bias/vAdam/sequential/p_re_lu/alpha/v!Adam/sequential/conv2d_1/kernel/vAdam/sequential/conv2d_1/bias/v!Adam/sequential/p_re_lu_1/alpha/v!Adam/sequential/conv2d_2/kernel/vAdam/sequential/conv2d_2/bias/v!Adam/sequential/p_re_lu_2/alpha/vAdam/sequential/dense/kernel/vAdam/sequential/dense/bias/v*-
_gradient_op_typePartitionedCall-119743*+
f&R$
"__inference__traced_restore_119742*
Tout
2**
config_proto

GPU 

CPU2J 8*4
Tin-
+2)*
_output_shapes
: ��
�4
�
C__inference_sequential_layer_call_and_return_conditional_losses_816

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2*
&p_re_lu_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2,
(p_re_lu_1_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2,
(p_re_lu_2_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�p_re_lu/StatefulPartitionedCall�!p_re_lu_1/StatefulPartitionedCall�!p_re_lu_2/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*0
_output_shapes
:���������:� **
_gradient_op_typePartitionedCall-122*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_121*
Tout
2**
config_proto

CPU

GPU 2J 8�
p_re_lu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0&p_re_lu_statefulpartitionedcall_args_1**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:���������:� **
_gradient_op_typePartitionedCall-162*I
fDRB
@__inference_p_re_lu_layer_call_and_return_conditional_losses_161*
Tout
2�
max_pooling2d/PartitionedCallPartitionedCall(p_re_lu/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:���������O *
Tin
2**
_gradient_op_typePartitionedCall-261*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_260*
Tout
2�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2**
_gradient_op_typePartitionedCall-647*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_646*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:���������K@*
Tin
2�
!p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0(p_re_lu_1_statefulpartitionedcall_args_1*K
fFRD
B__inference_p_re_lu_1_layer_call_and_return_conditional_losses_181*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������K@**
_gradient_op_typePartitionedCall-182�
max_pooling2d_1/PartitionedCallPartitionedCall*p_re_lu_1/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������%@**
_gradient_op_typePartitionedCall-286*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_285*
Tout
2�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:���������!�**
_gradient_op_typePartitionedCall-141*J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_140*
Tout
2�
!p_re_lu_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0(p_re_lu_2_statefulpartitionedcall_args_1**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:���������!�**
_gradient_op_typePartitionedCall-603*K
fFRD
B__inference_p_re_lu_2_layer_call_and_return_conditional_losses_602*
Tout
2�
max_pooling2d_2/PartitionedCallPartitionedCall*p_re_lu_2/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*0
_output_shapes
:����������*
Tin
2**
_gradient_op_typePartitionedCall-661*Q
fLRJ
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_660*
Tout
2�
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*(
_output_shapes
:����������@*
Tin
2**
_gradient_op_typePartitionedCall-274*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_273�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
_gradient_op_typePartitionedCall-103*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_102*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:�����������
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*I
fDRB
@__inference_reshape_layer_call_and_return_conditional_losses_629*
Tout
2**
config_proto

CPU

GPU 2J 8*+
_output_shapes
:���������>*
Tin
2**
_gradient_op_typePartitionedCall-630�
softmax/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*+
_output_shapes
:���������>*
Tin
2*(
_gradient_op_typePartitionedCall-7*G
fBR@
>__inference_softmax_layer_call_and_return_conditional_losses_6*
Tout
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentity softmax/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^p_re_lu/StatefulPartitionedCall"^p_re_lu_1/StatefulPartitionedCall"^p_re_lu_2/StatefulPartitionedCall*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*[
_input_shapesJ
H:���������<�:::::::::::2F
!p_re_lu_1/StatefulPartitionedCall!p_re_lu_1/StatefulPartitionedCall2F
!p_re_lu_2/StatefulPartitionedCall!p_re_lu_2/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
p_re_lu/StatefulPartitionedCallp_re_lu/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : 
�
�
$__inference_conv2d_layer_call_fn_129

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*A
_output_shapes/
-:+��������������������������� *
Tin
2**
_gradient_op_typePartitionedCall-122*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_121*
Tout
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+��������������������������� *
T0"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
|
E__forward_max_pooling2d_layer_call_and_return_conditional_losses_3921
inputs_0
identity

inputs
maxpool�
MaxPoolMaxPoolinputs_0*
ksize
*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0"
inputsinputs_0"
maxpoolMaxPool:output:0*s
backward_function_nameYW__inference___backward_max_pooling2d_layer_call_and_return_conditional_losses_3913_3922*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�

�
?__forward_p_re_lu_layer_call_and_return_conditional_losses_3973

inputs
readvariableop_resource
identity
relu
mul
neg

relu_1��ReadVariableOpi
ReluReluinputs*
T0*J
_output_shapes8
6:4�������������������������������������
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
::� P
NegNegReadVariableOp:value:0*
T0*#
_output_shapes
::� i
Neg_1Neginputs*
T0*J
_output_shapes8
6:4������������������������������������n
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4������������������������������������4
mul_0MulNeg:y:0Relu_1:activations:0*
T0f
addAddV2Relu:activations:0	mul_0:z:0*
T0*0
_output_shapes
:���������:� i
IdentityIdentityadd:z:0^ReadVariableOp*
T0*0
_output_shapes
:���������:� "
negNeg:y:0"
reluRelu:activations:0"
identityIdentity:output:0"
mul	mul_0:z:0"
relu_1Relu_1:activations:0*m
backward_function_nameSQ__inference___backward_p_re_lu_layer_call_and_return_conditional_losses_3940_3974*M
_input_shapes<
::4������������������������������������:2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: 
�
C
'__inference_restored_function_body_1532

inputs
identity�
PartitionedCallPartitionedCallinputs**
_gradient_op_typePartitionedCall-261*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_260*
Tout
2**
config_proto

GPU 

CPU2J 8*�
_output_shapes�
�:4������������������������������������:4������������������������������������:4������������������������������������*
Tin
2�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
\
@__inference_flatten_layer_call_and_return_conditional_losses_273

inputs
identity^
Reshape/shapeConst*
valueB"����    *
dtype0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������@Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������@"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
A__forward_p_re_lu_2_layer_call_and_return_conditional_losses_3719

inputs
readvariableop_resource
identity
relu
mul
neg

relu_1��ReadVariableOpi
ReluReluinputs*
T0*J
_output_shapes8
6:4�������������������������������������
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:!�P
NegNegReadVariableOp:value:0*#
_output_shapes
:!�*
T0i
Neg_1Neginputs*
T0*J
_output_shapes8
6:4������������������������������������n
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4������������������������������������4
mul_0MulNeg:y:0Relu_1:activations:0*
T0f
addAddV2Relu:activations:0	mul_0:z:0*
T0*0
_output_shapes
:���������!�i
IdentityIdentityadd:z:0^ReadVariableOp*0
_output_shapes
:���������!�*
T0"
negNeg:y:0"
reluRelu:activations:0"
identityIdentity:output:0"
mul	mul_0:z:0"
relu_1Relu_1:activations:0*o
backward_function_nameUS__inference___backward_p_re_lu_2_layer_call_and_return_conditional_losses_3686_3720*M
_input_shapes<
::4������������������������������������:2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: 
�
�
+__inference_sequential_layer_call_fn_119422
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11*+
_gradient_op_typePartitionedCall-5073*0
f+R)
'__inference_restored_function_body_5072*
Tout
2**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:���������>*
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*[
_input_shapesJ
H:���������<�:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : 
�
�
@__forward_conv2d_1_layer_call_and_return_conditional_losses_3888
inputs_0"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

inputs
conv2d_readvariableop��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: @*
dtype0�
Conv2DConv2Dinputs_0Conv2D/ReadVariableOp:value:0*
paddingVALID*A
_output_shapes/
-:+���������������������������@*
T0*
strides
�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+���������������������������@*
T0�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@"
identityIdentity:output:0"6
conv2d_readvariableopConv2D/ReadVariableOp:value:0"
inputsinputs_0*n
backward_function_nameTR__inference___backward_conv2d_1_layer_call_and_return_conditional_losses_3874_3889*H
_input_shapes7
5:+��������������������������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
\
@__inference_flatten_layer_call_and_return_conditional_losses_195

inputs
identity^
Reshape/shapeConst*
valueB"����    *
dtype0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:����������@*
T0Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������@"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
d
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_660

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
strides
*
ksize
*
paddingVALID{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�

�
A__inference_conv2d_1_layer_call_and_return_conditional_losses_646

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*A
_output_shapes/
-:+���������������������������@*
T0�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+���������������������������@*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
I
-__inference_max_pooling2d_1_layer_call_fn_291

inputs
identity�
PartitionedCallPartitionedCallinputs**
_gradient_op_typePartitionedCall-286*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_285*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*J
_output_shapes8
6:4�������������������������������������
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
C
'__inference_restored_function_body_1629

inputs
identity�
PartitionedCallPartitionedCallinputs*I
fDRB
@__inference_reshape_layer_call_and_return_conditional_losses_521*
Tout
2**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:���������>*
Tin
2**
_gradient_op_typePartitionedCall-522d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
'__inference_restored_function_body_1576

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*�
_output_shapesp
n:,����������������������������:+���������������������������@:@�**
_gradient_op_typePartitionedCall-141*J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_140*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*B
_output_shapes0
.:,����������������������������*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
\
@__inference_softmax_layer_call_and_return_conditional_losses_615

inputs
identityP
SoftmaxSoftmaxinputs*+
_output_shapes
:���������>*
T0]
IdentityIdentitySoftmax:softmax:0*+
_output_shapes
:���������>*
T0"
identityIdentity:output:0**
_input_shapes
:���������>:& "
 
_user_specified_nameinputs
�
C
'__inference_restored_function_body_1638

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*+
_output_shapes
:���������>**
_gradient_op_typePartitionedCall-616*I
fDRB
@__inference_softmax_layer_call_and_return_conditional_losses_615*
Tout
2**
config_proto

GPU 

CPU2J 8d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0**
_input_shapes
:���������>:& "
 
_user_specified_nameinputs
�O
�
__inference__traced_save_119609
file_prefix7
3savev2_sequential_conv2d_kernel_read_readvariableop5
1savev2_sequential_conv2d_bias_read_readvariableop7
3savev2_sequential_p_re_lu_alpha_read_readvariableop9
5savev2_sequential_conv2d_1_kernel_read_readvariableop7
3savev2_sequential_conv2d_1_bias_read_readvariableop9
5savev2_sequential_p_re_lu_1_alpha_read_readvariableop9
5savev2_sequential_conv2d_2_kernel_read_readvariableop7
3savev2_sequential_conv2d_2_bias_read_readvariableop9
5savev2_sequential_p_re_lu_2_alpha_read_readvariableop6
2savev2_sequential_dense_kernel_read_readvariableop4
0savev2_sequential_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop>
:savev2_adam_sequential_conv2d_kernel_m_read_readvariableop<
8savev2_adam_sequential_conv2d_bias_m_read_readvariableop>
:savev2_adam_sequential_p_re_lu_alpha_m_read_readvariableop@
<savev2_adam_sequential_conv2d_1_kernel_m_read_readvariableop>
:savev2_adam_sequential_conv2d_1_bias_m_read_readvariableop@
<savev2_adam_sequential_p_re_lu_1_alpha_m_read_readvariableop@
<savev2_adam_sequential_conv2d_2_kernel_m_read_readvariableop>
:savev2_adam_sequential_conv2d_2_bias_m_read_readvariableop@
<savev2_adam_sequential_p_re_lu_2_alpha_m_read_readvariableop=
9savev2_adam_sequential_dense_kernel_m_read_readvariableop;
7savev2_adam_sequential_dense_bias_m_read_readvariableop>
:savev2_adam_sequential_conv2d_kernel_v_read_readvariableop<
8savev2_adam_sequential_conv2d_bias_v_read_readvariableop>
:savev2_adam_sequential_p_re_lu_alpha_v_read_readvariableop@
<savev2_adam_sequential_conv2d_1_kernel_v_read_readvariableop>
:savev2_adam_sequential_conv2d_1_bias_v_read_readvariableop@
<savev2_adam_sequential_p_re_lu_1_alpha_v_read_readvariableop@
<savev2_adam_sequential_conv2d_2_kernel_v_read_readvariableop>
:savev2_adam_sequential_conv2d_2_bias_v_read_readvariableop@
<savev2_adam_sequential_p_re_lu_2_alpha_v_read_readvariableop=
9savev2_adam_sequential_dense_kernel_v_read_readvariableop;
7savev2_adam_sequential_dense_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_6952debf59b442bcbdeac63aa65a1445/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
dtype0*
_output_shapes
: *
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�(B)layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-0/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer-1/alpha/.ATTRIBUTES/VARIABLE_VALUEB)layer-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-3/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer-4/alpha/.ATTRIBUTES/VARIABLE_VALUEB)layer-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-6/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer-7/alpha/.ATTRIBUTES/VARIABLE_VALUEB*layer-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB(layer-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer-1/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer-4/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer-7/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFlayer-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer-1/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer-4/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer-7/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFlayer-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:(�
SaveV2/shape_and_slicesConst"/device:CPU:0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_sequential_conv2d_kernel_read_readvariableop1savev2_sequential_conv2d_bias_read_readvariableop3savev2_sequential_p_re_lu_alpha_read_readvariableop5savev2_sequential_conv2d_1_kernel_read_readvariableop3savev2_sequential_conv2d_1_bias_read_readvariableop5savev2_sequential_p_re_lu_1_alpha_read_readvariableop5savev2_sequential_conv2d_2_kernel_read_readvariableop3savev2_sequential_conv2d_2_bias_read_readvariableop5savev2_sequential_p_re_lu_2_alpha_read_readvariableop2savev2_sequential_dense_kernel_read_readvariableop0savev2_sequential_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop:savev2_adam_sequential_conv2d_kernel_m_read_readvariableop8savev2_adam_sequential_conv2d_bias_m_read_readvariableop:savev2_adam_sequential_p_re_lu_alpha_m_read_readvariableop<savev2_adam_sequential_conv2d_1_kernel_m_read_readvariableop:savev2_adam_sequential_conv2d_1_bias_m_read_readvariableop<savev2_adam_sequential_p_re_lu_1_alpha_m_read_readvariableop<savev2_adam_sequential_conv2d_2_kernel_m_read_readvariableop:savev2_adam_sequential_conv2d_2_bias_m_read_readvariableop<savev2_adam_sequential_p_re_lu_2_alpha_m_read_readvariableop9savev2_adam_sequential_dense_kernel_m_read_readvariableop7savev2_adam_sequential_dense_bias_m_read_readvariableop:savev2_adam_sequential_conv2d_kernel_v_read_readvariableop8savev2_adam_sequential_conv2d_bias_v_read_readvariableop:savev2_adam_sequential_p_re_lu_alpha_v_read_readvariableop<savev2_adam_sequential_conv2d_1_kernel_v_read_readvariableop:savev2_adam_sequential_conv2d_1_bias_v_read_readvariableop<savev2_adam_sequential_p_re_lu_1_alpha_v_read_readvariableop<savev2_adam_sequential_conv2d_2_kernel_v_read_readvariableop:savev2_adam_sequential_conv2d_2_bias_v_read_readvariableop<savev2_adam_sequential_p_re_lu_2_alpha_v_read_readvariableop9savev2_adam_sequential_dense_kernel_v_read_readvariableop7savev2_adam_sequential_dense_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : ::� : @:@:K@:@�:�:!�:
�@�:�: : : : : : : : : ::� : @:@:K@:@�:�:!�:
�@�:�: : ::� : @:@:K@:@�:�:!�:
�@�:�: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : 
�
I
-__inference_max_pooling2d_2_layer_call_fn_666

inputs
identity�
PartitionedCallPartitionedCallinputs**
_gradient_op_typePartitionedCall-661*Q
fLRJ
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_660*
Tout
2**
config_proto

CPU

GPU 2J 8*J
_output_shapes8
6:4������������������������������������*
Tin
2�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
b
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_260

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
strides
*
ksize
*
paddingVALID{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�	
�
B__inference_p_re_lu_2_layer_call_and_return_conditional_losses_602

inputs
readvariableop_resource
identity��ReadVariableOpi
ReluReluinputs*
T0*J
_output_shapes8
6:4�������������������������������������
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:!�P
NegNegReadVariableOp:value:0*
T0*#
_output_shapes
:!�i
Neg_1Neginputs*
T0*J
_output_shapes8
6:4������������������������������������n
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4������������������������������������d
mulMulNeg:y:0Relu_1:activations:0*0
_output_shapes
:���������!�*
T0d
addAddV2Relu:activations:0mul:z:0*0
_output_shapes
:���������!�*
T0i
IdentityIdentityadd:z:0^ReadVariableOp*
T0*0
_output_shapes
:���������!�"
identityIdentity:output:0*M
_input_shapes<
::4������������������������������������:2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: 
�
�
'__inference_restored_function_body_1543

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*�
_output_shapesn
l:+���������������������������@:+��������������������������� : @**
_gradient_op_typePartitionedCall-647*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_646*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�
�
'__inference_restored_function_body_1618

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*F
fAR?
=__inference_dense_layer_call_and_return_conditional_losses_23*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:����������*)
_gradient_op_typePartitionedCall-24�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
�
'__inference_restored_function_body_5072

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11**
_gradient_op_typePartitionedCall-788*1
f,R*
(__inference_sequential_layer_call_fn_787*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������>�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*[
_input_shapesJ
H:���������<�:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : 
�
�
'__inference_restored_function_body_1555

inputs"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1**
_gradient_op_typePartitionedCall-182*K
fFRD
B__inference_p_re_lu_1_layer_call_and_return_conditional_losses_181*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*�
_output_shapes�
�:���������K@:4������������������������������������:���������K@:K@:4�������������������������������������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������K@"
identityIdentity:output:0*M
_input_shapes<
::4������������������������������������:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: 
�
�
#__inference_dense_layer_call_fn_110

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:����������**
_gradient_op_typePartitionedCall-103*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_102*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
C
'__inference_restored_function_body_1607

inputs
identity�
PartitionedCallPartitionedCallinputs**
_gradient_op_typePartitionedCall-196*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_195*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:����������@*
Tin
2a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������@"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�

�
?__inference_conv2d_layer_call_and_return_conditional_losses_121

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*A
_output_shapes/
-:+��������������������������� �
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� �
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+��������������������������� *
T0"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
'__inference_p_re_lu_1_layer_call_fn_188

inputs"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1**
_gradient_op_typePartitionedCall-182*K
fFRD
B__inference_p_re_lu_1_layer_call_and_return_conditional_losses_181*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:���������K@*
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������K@"
identityIdentity:output:0*M
_input_shapes<
::4������������������������������������:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: 
�
�
=__inference_dense_layer_call_and_return_conditional_losses_23

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
�@�j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
�
>__inference_dense_layer_call_and_return_conditional_losses_102

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
�@�j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:�w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:����������*
T0�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�

�
A__forward_p_re_lu_1_layer_call_and_return_conditional_losses_3846

inputs
readvariableop_resource
identity
relu
mul
neg

relu_1��ReadVariableOpi
ReluReluinputs*J
_output_shapes8
6:4������������������������������������*
T0�
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*"
_output_shapes
:K@O
NegNegReadVariableOp:value:0*"
_output_shapes
:K@*
T0i
Neg_1Neginputs*
T0*J
_output_shapes8
6:4������������������������������������n
Relu_1Relu	Neg_1:y:0*J
_output_shapes8
6:4������������������������������������*
T04
mul_0MulNeg:y:0Relu_1:activations:0*
T0e
addAddV2Relu:activations:0	mul_0:z:0*/
_output_shapes
:���������K@*
T0h
IdentityIdentityadd:z:0^ReadVariableOp*
T0*/
_output_shapes
:���������K@"
relu_1Relu_1:activations:0"
negNeg:y:0"
reluRelu:activations:0"
identityIdentity:output:0"
mul	mul_0:z:0*o
backward_function_nameUS__inference___backward_p_re_lu_1_layer_call_and_return_conditional_losses_3813_3847*M
_input_shapes<
::4������������������������������������:2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: 
�
�
'__inference_p_re_lu_2_layer_call_fn_609

inputs"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:���������!�**
_gradient_op_typePartitionedCall-603*K
fFRD
B__inference_p_re_lu_2_layer_call_and_return_conditional_losses_602*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������!�"
identityIdentity:output:0*M
_input_shapes<
::4������������������������������������:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: 
�

�
A__inference_conv2d_2_layer_call_and_return_conditional_losses_140

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*'
_output_shapes
:@��
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*B
_output_shapes0
.:,����������������������������*
T0�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*B
_output_shapes0
.:,����������������������������*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
�
'__inference_restored_function_body_1510

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*�
_output_shapesn
l:+��������������������������� :+���������������������������: **
_gradient_op_typePartitionedCall-122*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_121*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� "
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
C
'__inference_restored_function_body_1565

inputs
identity�
PartitionedCallPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*�
_output_shapes�
�:4������������������������������������:4������������������������������������:4������������������������������������*
Tin
2**
_gradient_op_typePartitionedCall-286*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_285*
Tout
2�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�1
�
F__inference_sequential_layer_call_and_return_conditional_losses_119405
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2*
&p_re_lu_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2,
(p_re_lu_1_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2,
(p_re_lu_2_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�p_re_lu/StatefulPartitionedCall�!p_re_lu_1/StatefulPartitionedCall�!p_re_lu_2/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*0
f+R)
'__inference_restored_function_body_1510*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:���������:� *+
_gradient_op_typePartitionedCall-1511�
p_re_lu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0&p_re_lu_statefulpartitionedcall_args_1*+
_gradient_op_typePartitionedCall-1523*0
f+R)
'__inference_restored_function_body_1522*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:���������:� �
max_pooling2d/PartitionedCallPartitionedCall(p_re_lu/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:���������O *
Tin
2*+
_gradient_op_typePartitionedCall-1533*0
f+R)
'__inference_restored_function_body_1532*
Tout
2�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*/
_output_shapes
:���������K@*+
_gradient_op_typePartitionedCall-1544*0
f+R)
'__inference_restored_function_body_1543*
Tout
2**
config_proto

GPU 

CPU2J 8�
!p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0(p_re_lu_1_statefulpartitionedcall_args_1*
Tin
2*/
_output_shapes
:���������K@*+
_gradient_op_typePartitionedCall-1556*0
f+R)
'__inference_restored_function_body_1555*
Tout
2**
config_proto

GPU 

CPU2J 8�
max_pooling2d_1/PartitionedCallPartitionedCall*p_re_lu_1/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:���������%@*
Tin
2*+
_gradient_op_typePartitionedCall-1566*0
f+R)
'__inference_restored_function_body_1565*
Tout
2�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1577*0
f+R)
'__inference_restored_function_body_1576*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:���������!��
!p_re_lu_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0(p_re_lu_2_statefulpartitionedcall_args_1*0
_output_shapes
:���������!�*
Tin
2*+
_gradient_op_typePartitionedCall-1589*0
f+R)
'__inference_restored_function_body_1588*
Tout
2**
config_proto

GPU 

CPU2J 8�
max_pooling2d_2/PartitionedCallPartitionedCall*p_re_lu_2/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-1599*0
f+R)
'__inference_restored_function_body_1598*
Tout
2�
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1608*0
f+R)
'__inference_restored_function_body_1607*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:����������@�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-1619*0
f+R)
'__inference_restored_function_body_1618*
Tout
2�
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*+
_output_shapes
:���������>*+
_gradient_op_typePartitionedCall-1630*0
f+R)
'__inference_restored_function_body_1629*
Tout
2**
config_proto

GPU 

CPU2J 8�
softmax/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:���������>*
Tin
2*+
_gradient_op_typePartitionedCall-1639*0
f+R)
'__inference_restored_function_body_1638*
Tout
2�
IdentityIdentity softmax/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^p_re_lu/StatefulPartitionedCall"^p_re_lu_1/StatefulPartitionedCall"^p_re_lu_2/StatefulPartitionedCall*+
_output_shapes
:���������>*
T0"
identityIdentity:output:0*[
_input_shapesJ
H:���������<�:::::::::::2F
!p_re_lu_1/StatefulPartitionedCall!p_re_lu_1/StatefulPartitionedCall2F
!p_re_lu_2/StatefulPartitionedCall!p_re_lu_2/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
p_re_lu/StatefulPartitionedCallp_re_lu/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : 
�
�
'__inference_restored_function_body_1522

inputs"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1**
_gradient_op_typePartitionedCall-162*I
fDRB
@__inference_p_re_lu_layer_call_and_return_conditional_losses_161*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*�
_output_shapes�
�:���������:� :4������������������������������������:���������:� ::� :4�������������������������������������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*0
_output_shapes
:���������:� *
T0"
identityIdentity:output:0*M
_input_shapes<
::4������������������������������������:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: 
�

\
@__inference_reshape_layer_call_and_return_conditional_losses_629

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: Q
Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :Q
Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value	B :>�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
_output_shapes
:*
T0h
ReshapeReshapeinputsReshape/shape:output:0*+
_output_shapes
:���������>*
T0\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
&__inference_conv2d_1_layer_call_fn_654

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*A
_output_shapes/
-:+���������������������������@**
_gradient_op_typePartitionedCall-647*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_646*
Tout
2**
config_proto

CPU

GPU 2J 8�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs
�
�
%__inference_p_re_lu_layer_call_fn_168

inputs"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1**
_gradient_op_typePartitionedCall-162*I
fDRB
@__inference_p_re_lu_layer_call_and_return_conditional_losses_161*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:���������:� �
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������:� "
identityIdentity:output:0*M
_input_shapes<
::4������������������������������������:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: 
�
G
+__inference_max_pooling2d_layer_call_fn_266

inputs
identity�
PartitionedCallPartitionedCallinputs**
_gradient_op_typePartitionedCall-261*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_260*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*J
_output_shapes8
6:4�������������������������������������
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4������������������������������������*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�8
�
!__inference__wrapped_model_119344
input_14
0sequential_conv2d_statefulpartitionedcall_args_14
0sequential_conv2d_statefulpartitionedcall_args_25
1sequential_p_re_lu_statefulpartitionedcall_args_16
2sequential_conv2d_1_statefulpartitionedcall_args_16
2sequential_conv2d_1_statefulpartitionedcall_args_27
3sequential_p_re_lu_1_statefulpartitionedcall_args_16
2sequential_conv2d_2_statefulpartitionedcall_args_16
2sequential_conv2d_2_statefulpartitionedcall_args_27
3sequential_p_re_lu_2_statefulpartitionedcall_args_13
/sequential_dense_statefulpartitionedcall_args_13
/sequential_dense_statefulpartitionedcall_args_2
identity��)sequential/conv2d/StatefulPartitionedCall�+sequential/conv2d_1/StatefulPartitionedCall�+sequential/conv2d_2/StatefulPartitionedCall�(sequential/dense/StatefulPartitionedCall�*sequential/p_re_lu/StatefulPartitionedCall�,sequential/p_re_lu_1/StatefulPartitionedCall�,sequential/p_re_lu_2/StatefulPartitionedCall�
)sequential/conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_10sequential_conv2d_statefulpartitionedcall_args_10sequential_conv2d_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:���������:� *+
_gradient_op_typePartitionedCall-1511*0
f+R)
'__inference_restored_function_body_1510�
*sequential/p_re_lu/StatefulPartitionedCallStatefulPartitionedCall2sequential/conv2d/StatefulPartitionedCall:output:01sequential_p_re_lu_statefulpartitionedcall_args_1*+
_gradient_op_typePartitionedCall-1523*0
f+R)
'__inference_restored_function_body_1522*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:���������:� �
(sequential/max_pooling2d/PartitionedCallPartitionedCall3sequential/p_re_lu/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*/
_output_shapes
:���������O *+
_gradient_op_typePartitionedCall-1533*0
f+R)
'__inference_restored_function_body_1532*
Tout
2�
+sequential/conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1sequential/max_pooling2d/PartitionedCall:output:02sequential_conv2d_1_statefulpartitionedcall_args_12sequential_conv2d_1_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1544*0
f+R)
'__inference_restored_function_body_1543*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*/
_output_shapes
:���������K@�
,sequential/p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall4sequential/conv2d_1/StatefulPartitionedCall:output:03sequential_p_re_lu_1_statefulpartitionedcall_args_1*+
_gradient_op_typePartitionedCall-1556*0
f+R)
'__inference_restored_function_body_1555*
Tout
2**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:���������K@*
Tin
2�
*sequential/max_pooling2d_1/PartitionedCallPartitionedCall5sequential/p_re_lu_1/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*/
_output_shapes
:���������%@*+
_gradient_op_typePartitionedCall-1566*0
f+R)
'__inference_restored_function_body_1565*
Tout
2�
+sequential/conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3sequential/max_pooling2d_1/PartitionedCall:output:02sequential_conv2d_2_statefulpartitionedcall_args_12sequential_conv2d_2_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1577*0
f+R)
'__inference_restored_function_body_1576*
Tout
2**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:���������!�*
Tin
2�
,sequential/p_re_lu_2/StatefulPartitionedCallStatefulPartitionedCall4sequential/conv2d_2/StatefulPartitionedCall:output:03sequential_p_re_lu_2_statefulpartitionedcall_args_1*
Tin
2*0
_output_shapes
:���������!�*+
_gradient_op_typePartitionedCall-1589*0
f+R)
'__inference_restored_function_body_1588*
Tout
2**
config_proto

GPU 

CPU2J 8�
*sequential/max_pooling2d_2/PartitionedCallPartitionedCall5sequential/p_re_lu_2/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1599*0
f+R)
'__inference_restored_function_body_1598*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:�����������
"sequential/flatten/PartitionedCallPartitionedCall3sequential/max_pooling2d_2/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1608*0
f+R)
'__inference_restored_function_body_1607*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:����������@*
Tin
2�
(sequential/dense/StatefulPartitionedCallStatefulPartitionedCall+sequential/flatten/PartitionedCall:output:0/sequential_dense_statefulpartitionedcall_args_1/sequential_dense_statefulpartitionedcall_args_2*(
_output_shapes
:����������*
Tin
2*+
_gradient_op_typePartitionedCall-1619*0
f+R)
'__inference_restored_function_body_1618*
Tout
2**
config_proto

GPU 

CPU2J 8�
"sequential/reshape/PartitionedCallPartitionedCall1sequential/dense/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1630*0
f+R)
'__inference_restored_function_body_1629*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������>�
"sequential/softmax/PartitionedCallPartitionedCall+sequential/reshape/PartitionedCall:output:0*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������>*+
_gradient_op_typePartitionedCall-1639*0
f+R)
'__inference_restored_function_body_1638�
IdentityIdentity+sequential/softmax/PartitionedCall:output:0*^sequential/conv2d/StatefulPartitionedCall,^sequential/conv2d_1/StatefulPartitionedCall,^sequential/conv2d_2/StatefulPartitionedCall)^sequential/dense/StatefulPartitionedCall+^sequential/p_re_lu/StatefulPartitionedCall-^sequential/p_re_lu_1/StatefulPartitionedCall-^sequential/p_re_lu_2/StatefulPartitionedCall*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*[
_input_shapesJ
H:���������<�:::::::::::2T
(sequential/dense/StatefulPartitionedCall(sequential/dense/StatefulPartitionedCall2Z
+sequential/conv2d_1/StatefulPartitionedCall+sequential/conv2d_1/StatefulPartitionedCall2Z
+sequential/conv2d_2/StatefulPartitionedCall+sequential/conv2d_2/StatefulPartitionedCall2V
)sequential/conv2d/StatefulPartitionedCall)sequential/conv2d/StatefulPartitionedCall2\
,sequential/p_re_lu_1/StatefulPartitionedCall,sequential/p_re_lu_1/StatefulPartitionedCall2\
,sequential/p_re_lu_2/StatefulPartitionedCall,sequential/p_re_lu_2/StatefulPartitionedCall2X
*sequential/p_re_lu/StatefulPartitionedCall*sequential/p_re_lu/StatefulPartitionedCall: : : : : : : : :	 :
 : :' #
!
_user_specified_name	input_1
�

\
@__inference_reshape_layer_call_and_return_conditional_losses_521

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskQ
Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: Q
Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value	B :>�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
T0*
N*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:���������>\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
@__forward_conv2d_2_layer_call_and_return_conditional_losses_3761
inputs_0"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

inputs
conv2d_readvariableop��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*'
_output_shapes
:@��
Conv2DConv2Dinputs_0Conv2D/ReadVariableOp:value:0*B
_output_shapes0
.:,����������������������������*
T0*
strides
*
paddingVALID�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,����������������������������"6
conv2d_readvariableopConv2D/ReadVariableOp:value:0"
inputsinputs_0"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::*n
backward_function_nameTR__inference___backward_conv2d_2_layer_call_and_return_conditional_losses_3747_376220
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
&__inference_conv2d_2_layer_call_fn_148

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_140*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*B
_output_shapes0
.:,����������������������������**
_gradient_op_typePartitionedCall-141�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*B
_output_shapes0
.:,����������������������������*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
��
�
"__inference__traced_restore_119742
file_prefix-
)assignvariableop_sequential_conv2d_kernel-
)assignvariableop_1_sequential_conv2d_bias/
+assignvariableop_2_sequential_p_re_lu_alpha1
-assignvariableop_3_sequential_conv2d_1_kernel/
+assignvariableop_4_sequential_conv2d_1_bias1
-assignvariableop_5_sequential_p_re_lu_1_alpha1
-assignvariableop_6_sequential_conv2d_2_kernel/
+assignvariableop_7_sequential_conv2d_2_bias1
-assignvariableop_8_sequential_p_re_lu_2_alpha.
*assignvariableop_9_sequential_dense_kernel-
)assignvariableop_10_sequential_dense_bias!
assignvariableop_11_adam_iter#
assignvariableop_12_adam_beta_1#
assignvariableop_13_adam_beta_2"
assignvariableop_14_adam_decay*
&assignvariableop_15_adam_learning_rate
assignvariableop_16_total
assignvariableop_17_count7
3assignvariableop_18_adam_sequential_conv2d_kernel_m5
1assignvariableop_19_adam_sequential_conv2d_bias_m7
3assignvariableop_20_adam_sequential_p_re_lu_alpha_m9
5assignvariableop_21_adam_sequential_conv2d_1_kernel_m7
3assignvariableop_22_adam_sequential_conv2d_1_bias_m9
5assignvariableop_23_adam_sequential_p_re_lu_1_alpha_m9
5assignvariableop_24_adam_sequential_conv2d_2_kernel_m7
3assignvariableop_25_adam_sequential_conv2d_2_bias_m9
5assignvariableop_26_adam_sequential_p_re_lu_2_alpha_m6
2assignvariableop_27_adam_sequential_dense_kernel_m4
0assignvariableop_28_adam_sequential_dense_bias_m7
3assignvariableop_29_adam_sequential_conv2d_kernel_v5
1assignvariableop_30_adam_sequential_conv2d_bias_v7
3assignvariableop_31_adam_sequential_p_re_lu_alpha_v9
5assignvariableop_32_adam_sequential_conv2d_1_kernel_v7
3assignvariableop_33_adam_sequential_conv2d_1_bias_v9
5assignvariableop_34_adam_sequential_p_re_lu_1_alpha_v9
5assignvariableop_35_adam_sequential_conv2d_2_kernel_v7
3assignvariableop_36_adam_sequential_conv2d_2_bias_v9
5assignvariableop_37_adam_sequential_p_re_lu_2_alpha_v6
2assignvariableop_38_adam_sequential_dense_kernel_v4
0assignvariableop_39_adam_sequential_dense_bias_v
identity_41��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*�
value�B�(B)layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-0/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer-1/alpha/.ATTRIBUTES/VARIABLE_VALUEB)layer-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-3/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer-4/alpha/.ATTRIBUTES/VARIABLE_VALUEB)layer-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-6/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer-7/alpha/.ATTRIBUTES/VARIABLE_VALUEB*layer-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB(layer-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer-1/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer-4/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer-7/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFlayer-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDlayer-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer-1/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer-4/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer-7/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFlayer-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDlayer-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0�
RestoreV2/shape_and_slicesConst"/device:CPU:0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp)assignvariableop_sequential_conv2d_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp)assignvariableop_1_sequential_conv2d_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0�
AssignVariableOp_2AssignVariableOp+assignvariableop_2_sequential_p_re_lu_alphaIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0�
AssignVariableOp_3AssignVariableOp-assignvariableop_3_sequential_conv2d_1_kernelIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp+assignvariableop_4_sequential_conv2d_1_biasIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp-assignvariableop_5_sequential_p_re_lu_1_alphaIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0�
AssignVariableOp_6AssignVariableOp-assignvariableop_6_sequential_conv2d_2_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0�
AssignVariableOp_7AssignVariableOp+assignvariableop_7_sequential_conv2d_2_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0�
AssignVariableOp_8AssignVariableOp-assignvariableop_8_sequential_p_re_lu_2_alphaIdentity_8:output:0*
_output_shapes
 *
dtype0N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp*assignvariableop_9_sequential_dense_kernelIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp)assignvariableop_10_sequential_dense_biasIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0	
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_iterIdentity_11:output:0*
dtype0	*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_1Identity_12:output:0*
_output_shapes
 *
dtype0P
Identity_13IdentityRestoreV2:tensors:13*
_output_shapes
:*
T0�
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_2Identity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0�
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_decayIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp&assignvariableop_15_adam_learning_rateIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:{
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0*
_output_shapes
 *
dtype0P
Identity_17IdentityRestoreV2:tensors:17*
_output_shapes
:*
T0{
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0*
_output_shapes
 *
dtype0P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp3assignvariableop_18_adam_sequential_conv2d_kernel_mIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp1assignvariableop_19_adam_sequential_conv2d_bias_mIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp3assignvariableop_20_adam_sequential_p_re_lu_alpha_mIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T0�
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_sequential_conv2d_1_kernel_mIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0�
AssignVariableOp_22AssignVariableOp3assignvariableop_22_adam_sequential_conv2d_1_bias_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp5assignvariableop_23_adam_sequential_p_re_lu_1_alpha_mIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
_output_shapes
:*
T0�
AssignVariableOp_24AssignVariableOp5assignvariableop_24_adam_sequential_conv2d_2_kernel_mIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp3assignvariableop_25_adam_sequential_conv2d_2_bias_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
_output_shapes
:*
T0�
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adam_sequential_p_re_lu_2_alpha_mIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
_output_shapes
:*
T0�
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_sequential_dense_kernel_mIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
_output_shapes
:*
T0�
AssignVariableOp_28AssignVariableOp0assignvariableop_28_adam_sequential_dense_bias_mIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0�
AssignVariableOp_29AssignVariableOp3assignvariableop_29_adam_sequential_conv2d_kernel_vIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp1assignvariableop_30_adam_sequential_conv2d_bias_vIdentity_30:output:0*
_output_shapes
 *
dtype0P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp3assignvariableop_31_adam_sequential_p_re_lu_alpha_vIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adam_sequential_conv2d_1_kernel_vIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
_output_shapes
:*
T0�
AssignVariableOp_33AssignVariableOp3assignvariableop_33_adam_sequential_conv2d_1_bias_vIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp5assignvariableop_34_adam_sequential_p_re_lu_1_alpha_vIdentity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp5assignvariableop_35_adam_sequential_conv2d_2_kernel_vIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
_output_shapes
:*
T0�
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_sequential_conv2d_2_bias_vIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp5assignvariableop_37_adam_sequential_p_re_lu_2_alpha_vIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adam_sequential_dense_kernel_vIdentity_38:output:0*
dtype0*
_output_shapes
 P
Identity_39IdentityRestoreV2:tensors:39*
_output_shapes
:*
T0�
AssignVariableOp_39AssignVariableOp0assignvariableop_39_adam_sequential_dense_bias_vIdentity_39:output:0*
dtype0*
_output_shapes
 �
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: �
Identity_41IdentityIdentity_40:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_41Identity_41:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_32AssignVariableOp_322$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_39:" :# :$ :% :& :' :( :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! 
�
�
+__inference_sequential_layer_call_fn_119440
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������>*+
_gradient_op_typePartitionedCall-5108*0
f+R)
'__inference_restored_function_body_5107�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*[
_input_shapesJ
H:���������<�:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : 
�
�
>__forward_conv2d_layer_call_and_return_conditional_losses_4015
inputs_0"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity

inputs
conv2d_readvariableop��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: �
Conv2DConv2Dinputs_0Conv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+��������������������������� *
T0*
strides
*
paddingVALID�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� �
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� "
identityIdentity:output:0"6
conv2d_readvariableopConv2D/ReadVariableOp:value:0"
inputsinputs_0*l
backward_function_nameRP__inference___backward_conv2d_layer_call_and_return_conditional_losses_4001_4016*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
�
(__inference_sequential_layer_call_fn_787
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11**
config_proto

CPU

GPU 2J 8*+
_output_shapes
:���������>*
Tin
2**
_gradient_op_typePartitionedCall-754*L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_753*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*[
_input_shapesJ
H:���������<�:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:
 : :' #
!
_user_specified_name	input_1: : : : : : : : :	 
�
Z
>__inference_softmax_layer_call_and_return_conditional_losses_6

inputs
identityP
SoftmaxSoftmaxinputs*+
_output_shapes
:���������>*
T0]
IdentityIdentitySoftmax:softmax:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0**
_input_shapes
:���������>:& "
 
_user_specified_nameinputs
�
A
%__inference_flatten_layer_call_fn_279

inputs
identity�
PartitionedCallPartitionedCallinputs**
_gradient_op_typePartitionedCall-274*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_273*
Tout
2**
config_proto

CPU

GPU 2J 8*(
_output_shapes
:����������@*
Tin
2a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������@"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
d
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_285

inputs
identity�
MaxPoolMaxPoolinputs*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4������������������������������������{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
~
G__forward_max_pooling2d_1_layer_call_and_return_conditional_losses_3794
inputs_0
identity

inputs
maxpool�
MaxPoolMaxPoolinputs_0*J
_output_shapes8
6:4������������������������������������*
strides
*
ksize
*
paddingVALID{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
inputsinputs_0"
maxpoolMaxPool:output:0"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������*u
backward_function_name[Y__inference___backward_max_pooling2d_1_layer_call_and_return_conditional_losses_3786_3795:& "
 
_user_specified_nameinputs
�
�
'__inference_restored_function_body_1588

inputs"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*
Tout
2**
config_proto

GPU 

CPU2J 8*�
_output_shapes�
�:���������!�:4������������������������������������:���������!�:!�:4������������������������������������*
Tin
2**
_gradient_op_typePartitionedCall-603*K
fFRD
B__inference_p_re_lu_2_layer_call_and_return_conditional_losses_602�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:���������!�"
identityIdentity:output:0*M
_input_shapes<
::4������������������������������������:22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs
�
C
'__inference_restored_function_body_1598

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tout
2**
config_proto

GPU 

CPU2J 8*�
_output_shapes�
�:4������������������������������������:4������������������������������������:4������������������������������������*
Tin
2**
_gradient_op_typePartitionedCall-661*Q
fLRJ
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_660�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
A
%__inference_reshape_layer_call_fn_635

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tout
2**
config_proto

CPU

GPU 2J 8*+
_output_shapes
:���������>*
Tin
2**
_gradient_op_typePartitionedCall-630*I
fDRB
@__inference_reshape_layer_call_and_return_conditional_losses_629d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
'__inference_restored_function_body_5107

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11**
_gradient_op_typePartitionedCall-851*1
f,R*
(__inference_sequential_layer_call_fn_850*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������>�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*[
_input_shapesJ
H:���������<�:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : 
�
~
G__forward_max_pooling2d_2_layer_call_and_return_conditional_losses_3667
inputs_0
identity

inputs
maxpool�
MaxPoolMaxPoolinputs_0*J
_output_shapes8
6:4������������������������������������*
strides
*
ksize
*
paddingVALID{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
maxpoolMaxPool:output:0"
identityIdentity:output:0"
inputsinputs_0*u
backward_function_name[Y__inference___backward_max_pooling2d_2_layer_call_and_return_conditional_losses_3659_3668*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�1
�
F__inference_sequential_layer_call_and_return_conditional_losses_119375
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2*
&p_re_lu_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2,
(p_re_lu_1_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2,
(p_re_lu_2_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�p_re_lu/StatefulPartitionedCall�!p_re_lu_1/StatefulPartitionedCall�!p_re_lu_2/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1511*0
f+R)
'__inference_restored_function_body_1510*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:���������:� �
p_re_lu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0&p_re_lu_statefulpartitionedcall_args_1*+
_gradient_op_typePartitionedCall-1523*0
f+R)
'__inference_restored_function_body_1522*
Tout
2**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:���������:� *
Tin
2�
max_pooling2d/PartitionedCallPartitionedCall(p_re_lu/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1533*0
f+R)
'__inference_restored_function_body_1532*
Tout
2**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:���������O *
Tin
2�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*/
_output_shapes
:���������K@*+
_gradient_op_typePartitionedCall-1544*0
f+R)
'__inference_restored_function_body_1543�
!p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0(p_re_lu_1_statefulpartitionedcall_args_1**
config_proto

GPU 

CPU2J 8*
Tin
2*/
_output_shapes
:���������K@*+
_gradient_op_typePartitionedCall-1556*0
f+R)
'__inference_restored_function_body_1555*
Tout
2�
max_pooling2d_1/PartitionedCallPartitionedCall*p_re_lu_1/StatefulPartitionedCall:output:0*0
f+R)
'__inference_restored_function_body_1565*
Tout
2**
config_proto

GPU 

CPU2J 8*/
_output_shapes
:���������%@*
Tin
2*+
_gradient_op_typePartitionedCall-1566�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1577*0
f+R)
'__inference_restored_function_body_1576*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:���������!��
!p_re_lu_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0(p_re_lu_2_statefulpartitionedcall_args_1**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:���������!�*+
_gradient_op_typePartitionedCall-1589*0
f+R)
'__inference_restored_function_body_1588*
Tout
2�
max_pooling2d_2/PartitionedCallPartitionedCall*p_re_lu_2/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1599*0
f+R)
'__inference_restored_function_body_1598*
Tout
2**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:����������*
Tin
2�
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*0
f+R)
'__inference_restored_function_body_1607*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:����������@*
Tin
2*+
_gradient_op_typePartitionedCall-1608�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-1619*0
f+R)
'__inference_restored_function_body_1618*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:�����������
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1630*0
f+R)
'__inference_restored_function_body_1629*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������>�
softmax/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-1639*0
f+R)
'__inference_restored_function_body_1638*
Tout
2**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:���������>*
Tin
2�
IdentityIdentity softmax/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^p_re_lu/StatefulPartitionedCall"^p_re_lu_1/StatefulPartitionedCall"^p_re_lu_2/StatefulPartitionedCall*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*[
_input_shapesJ
H:���������<�:::::::::::2B
p_re_lu/StatefulPartitionedCallp_re_lu/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2F
!p_re_lu_1/StatefulPartitionedCall!p_re_lu_1/StatefulPartitionedCall2F
!p_re_lu_2/StatefulPartitionedCall!p_re_lu_2/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:
 : :' #
!
_user_specified_name	input_1: : : : : : : : :	 
�	
�
@__inference_p_re_lu_layer_call_and_return_conditional_losses_161

inputs
readvariableop_resource
identity��ReadVariableOpi
ReluReluinputs*J
_output_shapes8
6:4������������������������������������*
T0�
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
::� P
NegNegReadVariableOp:value:0*#
_output_shapes
::� *
T0i
Neg_1Neginputs*
T0*J
_output_shapes8
6:4������������������������������������n
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4������������������������������������d
mulMulNeg:y:0Relu_1:activations:0*
T0*0
_output_shapes
:���������:� d
addAddV2Relu:activations:0mul:z:0*0
_output_shapes
:���������:� *
T0i
IdentityIdentityadd:z:0^ReadVariableOp*
T0*0
_output_shapes
:���������:� "
identityIdentity:output:0*M
_input_shapes<
::4������������������������������������:2 
ReadVariableOpReadVariableOp: :& "
 
_user_specified_nameinputs
�
@
$__inference_softmax_layer_call_fn_12

inputs
identity�
PartitionedCallPartitionedCallinputs*(
_gradient_op_typePartitionedCall-7*G
fBR@
>__inference_softmax_layer_call_and_return_conditional_losses_6*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:���������>d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0**
_input_shapes
:���������>:& "
 
_user_specified_nameinputs
�4
�
C__inference_sequential_layer_call_and_return_conditional_losses_753

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2*
&p_re_lu_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2,
(p_re_lu_1_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2,
(p_re_lu_2_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�p_re_lu/StatefulPartitionedCall�!p_re_lu_1/StatefulPartitionedCall�!p_re_lu_2/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_121*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:���������:� **
_gradient_op_typePartitionedCall-122�
p_re_lu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0&p_re_lu_statefulpartitionedcall_args_1**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:���������:� **
_gradient_op_typePartitionedCall-162*I
fDRB
@__inference_p_re_lu_layer_call_and_return_conditional_losses_161*
Tout
2�
max_pooling2d/PartitionedCallPartitionedCall(p_re_lu/StatefulPartitionedCall:output:0**
_gradient_op_typePartitionedCall-261*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_260*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������O �
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_646*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:���������K@*
Tin
2**
_gradient_op_typePartitionedCall-647�
!p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0(p_re_lu_1_statefulpartitionedcall_args_1*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������K@**
_gradient_op_typePartitionedCall-182*K
fFRD
B__inference_p_re_lu_1_layer_call_and_return_conditional_losses_181�
max_pooling2d_1/PartitionedCallPartitionedCall*p_re_lu_1/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:���������%@**
_gradient_op_typePartitionedCall-286*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_285*
Tout
2�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2**
_gradient_op_typePartitionedCall-141*J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_140*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:���������!��
!p_re_lu_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0(p_re_lu_2_statefulpartitionedcall_args_1**
config_proto

CPU

GPU 2J 8*0
_output_shapes
:���������!�*
Tin
2**
_gradient_op_typePartitionedCall-603*K
fFRD
B__inference_p_re_lu_2_layer_call_and_return_conditional_losses_602*
Tout
2�
max_pooling2d_2/PartitionedCallPartitionedCall*p_re_lu_2/StatefulPartitionedCall:output:0**
_gradient_op_typePartitionedCall-661*Q
fLRJ
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_660*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:�����������
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:����������@**
_gradient_op_typePartitionedCall-274*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_273*
Tout
2�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
_gradient_op_typePartitionedCall-103*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_102*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:�����������
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*I
fDRB
@__inference_reshape_layer_call_and_return_conditional_losses_629*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:���������>**
_gradient_op_typePartitionedCall-630�
softmax/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*+
_output_shapes
:���������>*
Tin
2*(
_gradient_op_typePartitionedCall-7*G
fBR@
>__inference_softmax_layer_call_and_return_conditional_losses_6*
Tout
2�
IdentityIdentity softmax/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^p_re_lu/StatefulPartitionedCall"^p_re_lu_1/StatefulPartitionedCall"^p_re_lu_2/StatefulPartitionedCall*+
_output_shapes
:���������>*
T0"
identityIdentity:output:0*[
_input_shapesJ
H:���������<�:::::::::::2F
!p_re_lu_1/StatefulPartitionedCall!p_re_lu_1/StatefulPartitionedCall2F
!p_re_lu_2/StatefulPartitionedCall!p_re_lu_2/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
p_re_lu/StatefulPartitionedCallp_re_lu/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall: : : : : : : :	 :
 : :& "
 
_user_specified_nameinputs: 
�
�
(__inference_sequential_layer_call_fn_850

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11**
config_proto

CPU

GPU 2J 8*+
_output_shapes
:���������>*
Tin
2**
_gradient_op_typePartitionedCall-817*L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_816*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������>"
identityIdentity:output:0*[
_input_shapesJ
H:���������<�:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 
�	
�
B__inference_p_re_lu_1_layer_call_and_return_conditional_losses_181

inputs
readvariableop_resource
identity��ReadVariableOpi
ReluReluinputs*J
_output_shapes8
6:4������������������������������������*
T0�
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*"
_output_shapes
:K@O
NegNegReadVariableOp:value:0*
T0*"
_output_shapes
:K@i
Neg_1Neginputs*
T0*J
_output_shapes8
6:4������������������������������������n
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4������������������������������������c
mulMulNeg:y:0Relu_1:activations:0*
T0*/
_output_shapes
:���������K@c
addAddV2Relu:activations:0mul:z:0*
T0*/
_output_shapes
:���������K@h
IdentityIdentityadd:z:0^ReadVariableOp*
T0*/
_output_shapes
:���������K@"
identityIdentity:output:0*M
_input_shapes<
::4������������������������������������:2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: 
�
�
$__inference_signature_wrapper_119462
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11*-
_gradient_op_typePartitionedCall-119448**
f%R#
!__inference__wrapped_model_119344*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:���������>�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*+
_output_shapes
:���������>*
T0"
identityIdentity:output:0*[
_input_shapesJ
H:���������<�:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : : : :	 :
 : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
D
input_19
serving_default_input_1:0���������<�@
output_14
StatefulPartitionedCall:0���������>tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:��
�F
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
	optimizer

signatures
layer_with_weights-0
layer_with_weights-1
layer_with_weights-2
layer_with_weights-3
layer_with_weights-4
layer_with_weights-5
layer_with_weights-6
trainable_variables
regularization_losses
	variables
	keras_api
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"�A
_tf_keras_sequential�A{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "PReLU", "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "PReLU", "config": {"name": "p_re_lu_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "PReLU", "config": {"name": "p_re_lu_2", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 248, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [4, 62]}}, {"class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": -1}}], "build_input_shape": [null, 60, 160, 1]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "PReLU", "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "PReLU", "config": {"name": "p_re_lu_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "PReLU", "config": {"name": "p_re_lu_2", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 248, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [4, 62]}}, {"class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": -1}}], "build_input_shape": [null, 60, 160, 1]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
�
	alpha
trainable_variables
regularization_losses
	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "PReLU", "name": "p_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
trainable_variables
 regularization_losses
!	variables
"	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�

#kernel
$bias
%trainable_variables
&regularization_losses
'	variables
(	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
�
	)alpha
*trainable_variables
+regularization_losses
,	variables
-	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "PReLU", "name": "p_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "p_re_lu_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
.trainable_variables
/regularization_losses
0	variables
1	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�

2kernel
3bias
4trainable_variables
5regularization_losses
6	variables
7	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
�
	8alpha
9trainable_variables
:regularization_losses
;	variables
<	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "PReLU", "name": "p_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "p_re_lu_2", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
=trainable_variables
>regularization_losses
?	variables
@	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

Ekernel
Fbias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 248, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8192}}}}
�
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [4, 62]}}
�
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": -1}}
�
Siter

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_ratem�m�m�#m�$m�)m�2m�3m�8m�Em�Fm�v�v�v�#v�$v�)v�2v�3v�8v�Ev�Fv�"
	optimizer
-
�serving_default"
signature_map
n
0
1
2
#3
$4
)5
26
37
88
E9
F10"
trackable_list_wrapper
 "
trackable_list_wrapper
n
0
1
2
#3
$4
)5
26
37
88
E9
F10"
trackable_list_wrapper
�
Xmetrics
Ylayer_regularization_losses
Znon_trainable_variables
trainable_variables

[layers
regularization_losses
	variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
2:0 2sequential/conv2d/kernel
$:" 2sequential/conv2d/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
\metrics
]layer_regularization_losses
^non_trainable_variables

_layers
trainable_variables
regularization_losses
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
/:-:� 2sequential/p_re_lu/alpha
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
`metrics
alayer_regularization_losses
bnon_trainable_variables

clayers
trainable_variables
regularization_losses
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
dmetrics
elayer_regularization_losses
fnon_trainable_variables

glayers
trainable_variables
 regularization_losses
!	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
4:2 @2sequential/conv2d_1/kernel
&:$@2sequential/conv2d_1/bias
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
�
hmetrics
ilayer_regularization_losses
jnon_trainable_variables

klayers
%trainable_variables
&regularization_losses
'	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0:.K@2sequential/p_re_lu_1/alpha
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
)0"
trackable_list_wrapper
�
lmetrics
mlayer_regularization_losses
nnon_trainable_variables

olayers
*trainable_variables
+regularization_losses
,	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
pmetrics
qlayer_regularization_losses
rnon_trainable_variables

slayers
.trainable_variables
/regularization_losses
0	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
5:3@�2sequential/conv2d_2/kernel
':%�2sequential/conv2d_2/bias
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
�
tmetrics
ulayer_regularization_losses
vnon_trainable_variables

wlayers
4trainable_variables
5regularization_losses
6	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
1:/!�2sequential/p_re_lu_2/alpha
'
80"
trackable_list_wrapper
 "
trackable_list_wrapper
'
80"
trackable_list_wrapper
�
xmetrics
ylayer_regularization_losses
znon_trainable_variables

{layers
9trainable_variables
:regularization_losses
;	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
|metrics
}layer_regularization_losses
~non_trainable_variables

layers
=trainable_variables
>regularization_losses
?	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
Atrainable_variables
Bregularization_losses
C	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)
�@�2sequential/dense/kernel
$:"�2sequential/dense/bias
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
Gtrainable_variables
Hregularization_losses
I	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
Ktrainable_variables
Lregularization_losses
M	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
Otrainable_variables
Pregularization_losses
Q	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�
_fn_kwargs
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
�trainable_variables
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
7:5 2Adam/sequential/conv2d/kernel/m
):' 2Adam/sequential/conv2d/bias/m
4:2:� 2Adam/sequential/p_re_lu/alpha/m
9:7 @2!Adam/sequential/conv2d_1/kernel/m
+:)@2Adam/sequential/conv2d_1/bias/m
5:3K@2!Adam/sequential/p_re_lu_1/alpha/m
::8@�2!Adam/sequential/conv2d_2/kernel/m
,:*�2Adam/sequential/conv2d_2/bias/m
6:4!�2!Adam/sequential/p_re_lu_2/alpha/m
0:.
�@�2Adam/sequential/dense/kernel/m
):'�2Adam/sequential/dense/bias/m
7:5 2Adam/sequential/conv2d/kernel/v
):' 2Adam/sequential/conv2d/bias/v
4:2:� 2Adam/sequential/p_re_lu/alpha/v
9:7 @2!Adam/sequential/conv2d_1/kernel/v
+:)@2Adam/sequential/conv2d_1/bias/v
5:3K@2!Adam/sequential/p_re_lu_1/alpha/v
::8@�2!Adam/sequential/conv2d_2/kernel/v
,:*�2Adam/sequential/conv2d_2/bias/v
6:4!�2!Adam/sequential/p_re_lu_2/alpha/v
0:.
�@�2Adam/sequential/dense/kernel/v
):'�2Adam/sequential/dense/bias/v
�2�
F__inference_sequential_layer_call_and_return_conditional_losses_119405
F__inference_sequential_layer_call_and_return_conditional_losses_119375�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_119344�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� */�,
*�'
input_1���������<�
�2�
+__inference_sequential_layer_call_fn_119440
+__inference_sequential_layer_call_fn_119422�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_conv2d_layer_call_and_return_conditional_losses_121�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
$__inference_conv2d_layer_call_fn_129�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
@__inference_p_re_lu_layer_call_and_return_conditional_losses_161�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
%__inference_p_re_lu_layer_call_fn_168�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_260�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
+__inference_max_pooling2d_layer_call_fn_266�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
A__inference_conv2d_1_layer_call_and_return_conditional_losses_646�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
&__inference_conv2d_1_layer_call_fn_654�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
B__inference_p_re_lu_1_layer_call_and_return_conditional_losses_181�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
'__inference_p_re_lu_1_layer_call_fn_188�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_285�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
-__inference_max_pooling2d_1_layer_call_fn_291�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
A__inference_conv2d_2_layer_call_and_return_conditional_losses_140�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
&__inference_conv2d_2_layer_call_fn_148�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
B__inference_p_re_lu_2_layer_call_and_return_conditional_losses_602�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
'__inference_p_re_lu_2_layer_call_fn_609�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_660�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
-__inference_max_pooling2d_2_layer_call_fn_666�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
@__inference_flatten_layer_call_and_return_conditional_losses_195�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_flatten_layer_call_fn_279�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
=__inference_dense_layer_call_and_return_conditional_losses_23�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
#__inference_dense_layer_call_fn_110�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_reshape_layer_call_and_return_conditional_losses_521�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_reshape_layer_call_fn_635�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_softmax_layer_call_and_return_conditional_losses_615�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_softmax_layer_call_fn_12�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
3B1
$__inference_signature_wrapper_119462input_1
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_260�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
@__inference_reshape_layer_call_and_return_conditional_losses_521]0�-
&�#
!�
inputs����������
� ")�&
�
0���������>
� �
B__inference_p_re_lu_1_layer_call_and_return_conditional_losses_181�)R�O
H�E
C�@
inputs4������������������������������������
� "-�*
#� 
0���������K@
� �
-__inference_max_pooling2d_1_layer_call_fn_291�R�O
H�E
C�@
inputs4������������������������������������
� ";�84������������������������������������{
$__inference_softmax_layer_call_fn_12S3�0
)�&
$�!
inputs���������>
� "����������>�
F__inference_sequential_layer_call_and_return_conditional_losses_119405{#$)238EFA�>
7�4
*�'
input_1���������<�
p 

 
� ")�&
�
0���������>
� �
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_285�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
F__inference_sequential_layer_call_and_return_conditional_losses_119375{#$)238EFA�>
7�4
*�'
input_1���������<�
p

 
� ")�&
�
0���������>
� �
+__inference_sequential_layer_call_fn_119422n#$)238EFA�>
7�4
*�'
input_1���������<�
p

 
� "����������>y
%__inference_reshape_layer_call_fn_635P0�-
&�#
!�
inputs����������
� "����������>�
@__inference_softmax_layer_call_and_return_conditional_losses_615`3�0
)�&
$�!
inputs���������>
� ")�&
�
0���������>
� �
$__inference_signature_wrapper_119462�#$)238EFD�A
� 
:�7
5
input_1*�'
input_1���������<�"7�4
2
output_1&�#
output_1���������>�
?__inference_conv2d_layer_call_and_return_conditional_losses_121�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+��������������������������� 
� �
=__inference_dense_layer_call_and_return_conditional_losses_23^EF0�-
&�#
!�
inputs����������@
� "&�#
�
0����������
� �
+__inference_sequential_layer_call_fn_119440n#$)238EFA�>
7�4
*�'
input_1���������<�
p 

 
� "����������>�
&__inference_conv2d_1_layer_call_fn_654�#$I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+���������������������������@�
'__inference_p_re_lu_2_layer_call_fn_609z8R�O
H�E
C�@
inputs4������������������������������������
� "!����������!�x
#__inference_dense_layer_call_fn_110QEF0�-
&�#
!�
inputs����������@
� "������������
+__inference_max_pooling2d_layer_call_fn_266�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
A__inference_conv2d_1_layer_call_and_return_conditional_losses_646�#$I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������@
� �
@__inference_p_re_lu_layer_call_and_return_conditional_losses_161�R�O
H�E
C�@
inputs4������������������������������������
� ".�+
$�!
0���������:� 
� �
B__inference_p_re_lu_2_layer_call_and_return_conditional_losses_602�8R�O
H�E
C�@
inputs4������������������������������������
� ".�+
$�!
0���������!�
� �
$__inference_conv2d_layer_call_fn_129�I�F
?�<
:�7
inputs+���������������������������
� "2�/+��������������������������� �
%__inference_p_re_lu_layer_call_fn_168zR�O
H�E
C�@
inputs4������������������������������������
� "!����������:� �
!__inference__wrapped_model_119344�#$)238EF9�6
/�,
*�'
input_1���������<�
� "7�4
2
output_1&�#
output_1���������>�
'__inference_p_re_lu_1_layer_call_fn_188y)R�O
H�E
C�@
inputs4������������������������������������
� " ����������K@�
@__inference_flatten_layer_call_and_return_conditional_losses_195b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������@
� �
-__inference_max_pooling2d_2_layer_call_fn_666�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
A__inference_conv2d_2_layer_call_and_return_conditional_losses_140�23I�F
?�<
:�7
inputs+���������������������������@
� "@�=
6�3
0,����������������������������
� ~
%__inference_flatten_layer_call_fn_279U8�5
.�+
)�&
inputs����������
� "�����������@�
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_660�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
&__inference_conv2d_2_layer_call_fn_148�23I�F
?�<
:�7
inputs+���������������������������@
� "3�0,����������������������������