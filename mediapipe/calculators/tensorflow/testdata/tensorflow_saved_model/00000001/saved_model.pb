��
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
2
L2Loss
t"T
output"T"
Ttype:
2
�
LRN

input"T
output"T"
depth_radiusint"
biasfloat%  �?"
alphafloat%  �?"
betafloat%   ?"
Ttype0:
2
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
�
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �"serve*1.14.02v1.14.0-rc1-22-gaf24dc9��
r
Input_PlaceholderPlaceholder*
shape:*
dtype0*&
_output_shapes
:
�
0conv1/weights/Initializer/truncated_normal/shapeConst*%
valueB"         @   * 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
:
�
/conv1/weights/Initializer/truncated_normal/meanConst*
valueB
 *    * 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
: 
�
1conv1/weights/Initializer/truncated_normal/stddevConst*
valueB
 *��L=* 
_class
loc:@conv1/weights*
dtype0*
_output_shapes
: 
�
:conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal0conv1/weights/Initializer/truncated_normal/shape*
T0* 
_class
loc:@conv1/weights*
seed2 *
dtype0*&
_output_shapes
:@*

seed 
�
.conv1/weights/Initializer/truncated_normal/mulMul:conv1/weights/Initializer/truncated_normal/TruncatedNormal1conv1/weights/Initializer/truncated_normal/stddev* 
_class
loc:@conv1/weights*&
_output_shapes
:@*
T0
�
*conv1/weights/Initializer/truncated_normalAdd.conv1/weights/Initializer/truncated_normal/mul/conv1/weights/Initializer/truncated_normal/mean*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
:@
�
conv1/weights
VariableV2"/device:CPU:0*&
_output_shapes
:@*
shared_name * 
_class
loc:@conv1/weights*
	container *
shape:@*
dtype0
�
conv1/weights/AssignAssignconv1/weights*conv1/weights/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
:@
�
conv1/weights/readIdentityconv1/weights"/device:CPU:0*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
:@
�
conv1/Conv2DConv2DInput_Placeholderconv1/weights/read*&
_output_shapes
:@*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
�
conv1/biases/Initializer/ConstConst*
valueB@*    *
_class
loc:@conv1/biases*
dtype0*
_output_shapes
:@
�
conv1/biases
VariableV2"/device:CPU:0*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@conv1/biases*
	container *
shape:@
�
conv1/biases/AssignAssignconv1/biasesconv1/biases/Initializer/Const"/device:CPU:0*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(
�
conv1/biases/readIdentityconv1/biases"/device:CPU:0*
T0*
_class
loc:@conv1/biases*
_output_shapes
:@
�
conv1/BiasAddBiasAddconv1/Conv2Dconv1/biases/read*
T0*
data_formatNHWC*&
_output_shapes
:@
S
conv1/conv1Reluconv1/BiasAdd*
T0*&
_output_shapes
:@

!conv1/conv1/conv1/activations/tagConst*.
value%B# Bconv1/conv1/conv1/activations*
dtype0*
_output_shapes
: 
�
conv1/conv1/conv1/activationsHistogramSummary!conv1/conv1/conv1/activations/tagconv1/conv1*
T0*
_output_shapes
: 
\
conv1/zero_fraction/SizeConst*
valueB		 R��*
dtype0	*
_output_shapes
: 
e
conv1/zero_fraction/LessEqual/yConst*
_output_shapes
: *
valueB	 R����*
dtype0	
�
conv1/zero_fraction/LessEqual	LessEqualconv1/zero_fraction/Sizeconv1/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
�
conv1/zero_fraction/cond/SwitchSwitchconv1/zero_fraction/LessEqualconv1/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
q
!conv1/zero_fraction/cond/switch_tIdentity!conv1/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

o
!conv1/zero_fraction/cond/switch_fIdentityconv1/zero_fraction/cond/Switch*
_output_shapes
: *
T0

l
 conv1/zero_fraction/cond/pred_idIdentityconv1/zero_fraction/LessEqual*
_output_shapes
: *
T0

�
,conv1/zero_fraction/cond/count_nonzero/zerosConst"^conv1/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
�
/conv1/zero_fraction/cond/count_nonzero/NotEqualNotEqual8conv1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1,conv1/zero_fraction/cond/count_nonzero/zeros*&
_output_shapes
:@*
T0
�
6conv1/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchconv1/conv1 conv1/zero_fraction/cond/pred_id*
T0*
_class
loc:@conv1/conv1*8
_output_shapes&
$:@:@
�
+conv1/zero_fraction/cond/count_nonzero/CastCast/conv1/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
Truncate( *&
_output_shapes
:@*

DstT0
�
,conv1/zero_fraction/cond/count_nonzero/ConstConst"^conv1/zero_fraction/cond/switch_t*%
valueB"             *
dtype0*
_output_shapes
:
�
4conv1/zero_fraction/cond/count_nonzero/nonzero_countSum+conv1/zero_fraction/cond/count_nonzero/Cast,conv1/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
conv1/zero_fraction/cond/CastCast4conv1/zero_fraction/cond/count_nonzero/nonzero_count*
Truncate( *
_output_shapes
: *

DstT0	*

SrcT0
�
.conv1/zero_fraction/cond/count_nonzero_1/zerosConst"^conv1/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
�
1conv1/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual8conv1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch.conv1/zero_fraction/cond/count_nonzero_1/zeros*&
_output_shapes
:@*
T0
�
8conv1/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchconv1/conv1 conv1/zero_fraction/cond/pred_id*
T0*
_class
loc:@conv1/conv1*8
_output_shapes&
$:@:@
�
-conv1/zero_fraction/cond/count_nonzero_1/CastCast1conv1/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *&
_output_shapes
:@*

DstT0	
�
.conv1/zero_fraction/cond/count_nonzero_1/ConstConst"^conv1/zero_fraction/cond/switch_f*%
valueB"             *
dtype0*
_output_shapes
:
�
6conv1/zero_fraction/cond/count_nonzero_1/nonzero_countSum-conv1/zero_fraction/cond/count_nonzero_1/Cast.conv1/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0	
�
conv1/zero_fraction/cond/MergeMerge6conv1/zero_fraction/cond/count_nonzero_1/nonzero_countconv1/zero_fraction/cond/Cast*
_output_shapes
: : *
T0	*
N
�
*conv1/zero_fraction/counts_to_fraction/subSubconv1/zero_fraction/Sizeconv1/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
�
+conv1/zero_fraction/counts_to_fraction/CastCast*conv1/zero_fraction/counts_to_fraction/sub*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
�
-conv1/zero_fraction/counts_to_fraction/Cast_1Castconv1/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	*
Truncate( 
�
.conv1/zero_fraction/counts_to_fraction/truedivRealDiv+conv1/zero_fraction/counts_to_fraction/Cast-conv1/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
conv1/zero_fraction/fractionIdentity.conv1/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
z
conv1/conv1/conv1/sparsity/tagsConst*+
value"B  Bconv1/conv1/conv1/sparsity*
dtype0*
_output_shapes
: 
�
conv1/conv1/conv1/sparsityScalarSummaryconv1/conv1/conv1/sparsity/tagsconv1/zero_fraction/fraction*
_output_shapes
: *
T0
�
pool1MaxPoolconv1/conv1*&
_output_shapes
:@*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME
�
norm1LRNpool1*
beta%  @?*&
_output_shapes
:@*
depth_radius*
bias%  �?*
T0*
alpha%S�8
�
0conv2/weights/Initializer/truncated_normal/shapeConst*%
valueB"      @   @   * 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
:
�
/conv2/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    * 
_class
loc:@conv2/weights
�
1conv2/weights/Initializer/truncated_normal/stddevConst*
valueB
 *��L=* 
_class
loc:@conv2/weights*
dtype0*
_output_shapes
: 
�
:conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal0conv2/weights/Initializer/truncated_normal/shape*&
_output_shapes
:@@*

seed *
T0* 
_class
loc:@conv2/weights*
seed2 *
dtype0
�
.conv2/weights/Initializer/truncated_normal/mulMul:conv2/weights/Initializer/truncated_normal/TruncatedNormal1conv2/weights/Initializer/truncated_normal/stddev*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
:@@
�
*conv2/weights/Initializer/truncated_normalAdd.conv2/weights/Initializer/truncated_normal/mul/conv2/weights/Initializer/truncated_normal/mean*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
:@@
�
conv2/weights
VariableV2"/device:CPU:0* 
_class
loc:@conv2/weights*
	container *
shape:@@*
dtype0*&
_output_shapes
:@@*
shared_name 
�
conv2/weights/AssignAssignconv2/weights*conv2/weights/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
:@@
�
conv2/weights/readIdentityconv2/weights"/device:CPU:0*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
:@@
�
conv2/Conv2DConv2Dnorm1conv2/weights/read*&
_output_shapes
:@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
�
conv2/biases/Initializer/ConstConst*
valueB@*���=*
_class
loc:@conv2/biases*
dtype0*
_output_shapes
:@
�
conv2/biases
VariableV2"/device:CPU:0*
dtype0*
_output_shapes
:@*
shared_name *
_class
loc:@conv2/biases*
	container *
shape:@
�
conv2/biases/AssignAssignconv2/biasesconv2/biases/Initializer/Const"/device:CPU:0*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@
�
conv2/biases/readIdentityconv2/biases"/device:CPU:0*
T0*
_class
loc:@conv2/biases*
_output_shapes
:@
�
conv2/BiasAddBiasAddconv2/Conv2Dconv2/biases/read*
T0*
data_formatNHWC*&
_output_shapes
:@
S
conv2/conv2Reluconv2/BiasAdd*
T0*&
_output_shapes
:@

!conv2/conv2/conv2/activations/tagConst*
_output_shapes
: *.
value%B# Bconv2/conv2/conv2/activations*
dtype0
�
conv2/conv2/conv2/activationsHistogramSummary!conv2/conv2/conv2/activations/tagconv2/conv2*
T0*
_output_shapes
: 
[
conv2/zero_fraction/SizeConst*
_output_shapes
: *
value
B	 R�H*
dtype0	
e
conv2/zero_fraction/LessEqual/yConst*
valueB	 R����*
dtype0	*
_output_shapes
: 
�
conv2/zero_fraction/LessEqual	LessEqualconv2/zero_fraction/Sizeconv2/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
�
conv2/zero_fraction/cond/SwitchSwitchconv2/zero_fraction/LessEqualconv2/zero_fraction/LessEqual*
_output_shapes
: : *
T0

q
!conv2/zero_fraction/cond/switch_tIdentity!conv2/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
o
!conv2/zero_fraction/cond/switch_fIdentityconv2/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
l
 conv2/zero_fraction/cond/pred_idIdentityconv2/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
�
,conv2/zero_fraction/cond/count_nonzero/zerosConst"^conv2/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
/conv2/zero_fraction/cond/count_nonzero/NotEqualNotEqual8conv2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1,conv2/zero_fraction/cond/count_nonzero/zeros*
T0*&
_output_shapes
:@
�
6conv2/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchconv2/conv2 conv2/zero_fraction/cond/pred_id*8
_output_shapes&
$:@:@*
T0*
_class
loc:@conv2/conv2
�
+conv2/zero_fraction/cond/count_nonzero/CastCast/conv2/zero_fraction/cond/count_nonzero/NotEqual*
Truncate( *&
_output_shapes
:@*

DstT0*

SrcT0

�
,conv2/zero_fraction/cond/count_nonzero/ConstConst"^conv2/zero_fraction/cond/switch_t*%
valueB"             *
dtype0*
_output_shapes
:
�
4conv2/zero_fraction/cond/count_nonzero/nonzero_countSum+conv2/zero_fraction/cond/count_nonzero/Cast,conv2/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
conv2/zero_fraction/cond/CastCast4conv2/zero_fraction/cond/count_nonzero/nonzero_count*
Truncate( *
_output_shapes
: *

DstT0	*

SrcT0
�
.conv2/zero_fraction/cond/count_nonzero_1/zerosConst"^conv2/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
�
1conv2/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual8conv2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch.conv2/zero_fraction/cond/count_nonzero_1/zeros*&
_output_shapes
:@*
T0
�
8conv2/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchconv2/conv2 conv2/zero_fraction/cond/pred_id*
T0*
_class
loc:@conv2/conv2*8
_output_shapes&
$:@:@
�
-conv2/zero_fraction/cond/count_nonzero_1/CastCast1conv2/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *&
_output_shapes
:@*

DstT0	
�
.conv2/zero_fraction/cond/count_nonzero_1/ConstConst"^conv2/zero_fraction/cond/switch_f*%
valueB"             *
dtype0*
_output_shapes
:
�
6conv2/zero_fraction/cond/count_nonzero_1/nonzero_countSum-conv2/zero_fraction/cond/count_nonzero_1/Cast.conv2/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0	
�
conv2/zero_fraction/cond/MergeMerge6conv2/zero_fraction/cond/count_nonzero_1/nonzero_countconv2/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
�
*conv2/zero_fraction/counts_to_fraction/subSubconv2/zero_fraction/Sizeconv2/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
�
+conv2/zero_fraction/counts_to_fraction/CastCast*conv2/zero_fraction/counts_to_fraction/sub*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
�
-conv2/zero_fraction/counts_to_fraction/Cast_1Castconv2/zero_fraction/Size*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
�
.conv2/zero_fraction/counts_to_fraction/truedivRealDiv+conv2/zero_fraction/counts_to_fraction/Cast-conv2/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
conv2/zero_fraction/fractionIdentity.conv2/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
z
conv2/conv2/conv2/sparsity/tagsConst*
_output_shapes
: *+
value"B  Bconv2/conv2/conv2/sparsity*
dtype0
�
conv2/conv2/conv2/sparsityScalarSummaryconv2/conv2/conv2/sparsity/tagsconv2/zero_fraction/fraction*
T0*
_output_shapes
: 
�
norm2LRNconv2/conv2*
alpha%S�8*
beta%  @?*&
_output_shapes
:@*
depth_radius*
bias%  �?*
T0
�
pool2MaxPoolnorm2*
ksize
*
paddingSAME*&
_output_shapes
:@*
T0*
data_formatNHWC*
strides

e
local3/Reshape/shapeConst*
valueB"   ����*
dtype0*
_output_shapes
:
n
local3/ReshapeReshapepool2local3/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	�
�
1local3/weights/Initializer/truncated_normal/shapeConst*
valueB" 	  �  *!
_class
loc:@local3/weights*
dtype0*
_output_shapes
:
�
0local3/weights/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *!
_class
loc:@local3/weights*
dtype0
�
2local3/weights/Initializer/truncated_normal/stddevConst*
valueB
 *
�#=*!
_class
loc:@local3/weights*
dtype0*
_output_shapes
: 
�
;local3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal1local3/weights/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
��*

seed *
T0*!
_class
loc:@local3/weights
�
/local3/weights/Initializer/truncated_normal/mulMul;local3/weights/Initializer/truncated_normal/TruncatedNormal2local3/weights/Initializer/truncated_normal/stddev*!
_class
loc:@local3/weights* 
_output_shapes
:
��*
T0
�
+local3/weights/Initializer/truncated_normalAdd/local3/weights/Initializer/truncated_normal/mul0local3/weights/Initializer/truncated_normal/mean*
T0*!
_class
loc:@local3/weights* 
_output_shapes
:
��
�
local3/weights
VariableV2"/device:CPU:0*
shared_name *!
_class
loc:@local3/weights*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
local3/weights/AssignAssignlocal3/weights+local3/weights/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*!
_class
loc:@local3/weights*
validate_shape(* 
_output_shapes
:
��
�
local3/weights/readIdentitylocal3/weights"/device:CPU:0* 
_output_shapes
:
��*
T0*!
_class
loc:@local3/weights
M
local3/L2LossL2Losslocal3/weights/read*
_output_shapes
: *
T0
Y
local3/weight_loss/yConst*
valueB
 *o�;*
dtype0*
_output_shapes
: 
_
local3/weight_lossMullocal3/L2Losslocal3/weight_loss/y*
_output_shapes
: *
T0
�
local3/biases/Initializer/ConstConst*
valueB�*���=* 
_class
loc:@local3/biases*
dtype0*
_output_shapes	
:�
�
local3/biases
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:�*
shared_name * 
_class
loc:@local3/biases*
	container *
shape:�
�
local3/biases/AssignAssignlocal3/biaseslocal3/biases/Initializer/Const"/device:CPU:0*
use_locking(*
T0* 
_class
loc:@local3/biases*
validate_shape(*
_output_shapes	
:�
�
local3/biases/readIdentitylocal3/biases"/device:CPU:0*
_output_shapes	
:�*
T0* 
_class
loc:@local3/biases
�
local3/MatMulMatMullocal3/Reshapelocal3/weights/read*
T0*
_output_shapes
:	�*
transpose_a( *
transpose_b( 
^

local3/addAddlocal3/MatMullocal3/biases/read*
T0*
_output_shapes
:	�
K
local3/local3Relu
local3/add*
T0*
_output_shapes
:	�
�
$local3/local3/local3/activations/tagConst*1
value(B& B local3/local3/local3/activations*
dtype0*
_output_shapes
: 
�
 local3/local3/local3/activationsHistogramSummary$local3/local3/local3/activations/taglocal3/local3*
T0*
_output_shapes
: 
\
local3/zero_fraction/SizeConst*
value
B	 R�*
dtype0	*
_output_shapes
: 
f
 local3/zero_fraction/LessEqual/yConst*
valueB	 R����*
dtype0	*
_output_shapes
: 
�
local3/zero_fraction/LessEqual	LessEquallocal3/zero_fraction/Size local3/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
�
 local3/zero_fraction/cond/SwitchSwitchlocal3/zero_fraction/LessEquallocal3/zero_fraction/LessEqual*
_output_shapes
: : *
T0

s
"local3/zero_fraction/cond/switch_tIdentity"local3/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
q
"local3/zero_fraction/cond/switch_fIdentity local3/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
n
!local3/zero_fraction/cond/pred_idIdentitylocal3/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
�
-local3/zero_fraction/cond/count_nonzero/zerosConst#^local3/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
0local3/zero_fraction/cond/count_nonzero/NotEqualNotEqual9local3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1-local3/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes
:	�
�
7local3/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchlocal3/local3!local3/zero_fraction/cond/pred_id*
T0* 
_class
loc:@local3/local3**
_output_shapes
:	�:	�
�
,local3/zero_fraction/cond/count_nonzero/CastCast0local3/zero_fraction/cond/count_nonzero/NotEqual*
Truncate( *
_output_shapes
:	�*

DstT0*

SrcT0

�
-local3/zero_fraction/cond/count_nonzero/ConstConst#^local3/zero_fraction/cond/switch_t*
_output_shapes
:*
valueB"       *
dtype0
�
5local3/zero_fraction/cond/count_nonzero/nonzero_countSum,local3/zero_fraction/cond/count_nonzero/Cast-local3/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
local3/zero_fraction/cond/CastCast5local3/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0	
�
/local3/zero_fraction/cond/count_nonzero_1/zerosConst#^local3/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
�
2local3/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual9local3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch/local3/zero_fraction/cond/count_nonzero_1/zeros*
_output_shapes
:	�*
T0
�
9local3/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchlocal3/local3!local3/zero_fraction/cond/pred_id*
T0* 
_class
loc:@local3/local3**
_output_shapes
:	�:	�
�
.local3/zero_fraction/cond/count_nonzero_1/CastCast2local3/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *
_output_shapes
:	�*

DstT0	
�
/local3/zero_fraction/cond/count_nonzero_1/ConstConst#^local3/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
�
7local3/zero_fraction/cond/count_nonzero_1/nonzero_countSum.local3/zero_fraction/cond/count_nonzero_1/Cast/local3/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0	
�
local3/zero_fraction/cond/MergeMerge7local3/zero_fraction/cond/count_nonzero_1/nonzero_countlocal3/zero_fraction/cond/Cast*
N*
_output_shapes
: : *
T0	
�
+local3/zero_fraction/counts_to_fraction/subSublocal3/zero_fraction/Sizelocal3/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
�
,local3/zero_fraction/counts_to_fraction/CastCast+local3/zero_fraction/counts_to_fraction/sub*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
�
.local3/zero_fraction/counts_to_fraction/Cast_1Castlocal3/zero_fraction/Size*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
�
/local3/zero_fraction/counts_to_fraction/truedivRealDiv,local3/zero_fraction/counts_to_fraction/Cast.local3/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
{
local3/zero_fraction/fractionIdentity/local3/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
�
"local3/local3/local3/sparsity/tagsConst*.
value%B# Blocal3/local3/local3/sparsity*
dtype0*
_output_shapes
: 
�
local3/local3/local3/sparsityScalarSummary"local3/local3/local3/sparsity/tagslocal3/zero_fraction/fraction*
_output_shapes
: *
T0
�
1local4/weights/Initializer/truncated_normal/shapeConst*
valueB"�  �   *!
_class
loc:@local4/weights*
dtype0*
_output_shapes
:
�
0local4/weights/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *!
_class
loc:@local4/weights*
dtype0
�
2local4/weights/Initializer/truncated_normal/stddevConst*
valueB
 *
�#=*!
_class
loc:@local4/weights*
dtype0*
_output_shapes
: 
�
;local4/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal1local4/weights/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
��*

seed *
T0*!
_class
loc:@local4/weights*
seed2 
�
/local4/weights/Initializer/truncated_normal/mulMul;local4/weights/Initializer/truncated_normal/TruncatedNormal2local4/weights/Initializer/truncated_normal/stddev*
T0*!
_class
loc:@local4/weights* 
_output_shapes
:
��
�
+local4/weights/Initializer/truncated_normalAdd/local4/weights/Initializer/truncated_normal/mul0local4/weights/Initializer/truncated_normal/mean* 
_output_shapes
:
��*
T0*!
_class
loc:@local4/weights
�
local4/weights
VariableV2"/device:CPU:0* 
_output_shapes
:
��*
shared_name *!
_class
loc:@local4/weights*
	container *
shape:
��*
dtype0
�
local4/weights/AssignAssignlocal4/weights+local4/weights/Initializer/truncated_normal"/device:CPU:0*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*!
_class
loc:@local4/weights
�
local4/weights/readIdentitylocal4/weights"/device:CPU:0*
T0*!
_class
loc:@local4/weights* 
_output_shapes
:
��
M
local4/L2LossL2Losslocal4/weights/read*
T0*
_output_shapes
: 
Y
local4/weight_loss/yConst*
valueB
 *o�;*
dtype0*
_output_shapes
: 
_
local4/weight_lossMullocal4/L2Losslocal4/weight_loss/y*
T0*
_output_shapes
: 
�
local4/biases/Initializer/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*���=* 
_class
loc:@local4/biases
�
local4/biases
VariableV2"/device:CPU:0*
_output_shapes	
:�*
shared_name * 
_class
loc:@local4/biases*
	container *
shape:�*
dtype0
�
local4/biases/AssignAssignlocal4/biaseslocal4/biases/Initializer/Const"/device:CPU:0*
use_locking(*
T0* 
_class
loc:@local4/biases*
validate_shape(*
_output_shapes	
:�
�
local4/biases/readIdentitylocal4/biases"/device:CPU:0*
T0* 
_class
loc:@local4/biases*
_output_shapes	
:�
�
local4/MatMulMatMullocal3/local3local4/weights/read*
_output_shapes
:	�*
transpose_a( *
transpose_b( *
T0
^

local4/addAddlocal4/MatMullocal4/biases/read*
_output_shapes
:	�*
T0
K
local4/local4Relu
local4/add*
T0*
_output_shapes
:	�
�
$local4/local4/local4/activations/tagConst*
_output_shapes
: *1
value(B& B local4/local4/local4/activations*
dtype0
�
 local4/local4/local4/activationsHistogramSummary$local4/local4/local4/activations/taglocal4/local4*
_output_shapes
: *
T0
\
local4/zero_fraction/SizeConst*
value
B	 R�*
dtype0	*
_output_shapes
: 
f
 local4/zero_fraction/LessEqual/yConst*
valueB	 R����*
dtype0	*
_output_shapes
: 
�
local4/zero_fraction/LessEqual	LessEquallocal4/zero_fraction/Size local4/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
�
 local4/zero_fraction/cond/SwitchSwitchlocal4/zero_fraction/LessEquallocal4/zero_fraction/LessEqual*
_output_shapes
: : *
T0

s
"local4/zero_fraction/cond/switch_tIdentity"local4/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

q
"local4/zero_fraction/cond/switch_fIdentity local4/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
n
!local4/zero_fraction/cond/pred_idIdentitylocal4/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
�
-local4/zero_fraction/cond/count_nonzero/zerosConst#^local4/zero_fraction/cond/switch_t*
_output_shapes
: *
valueB
 *    *
dtype0
�
0local4/zero_fraction/cond/count_nonzero/NotEqualNotEqual9local4/zero_fraction/cond/count_nonzero/NotEqual/Switch:1-local4/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes
:	�
�
7local4/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchlocal4/local4!local4/zero_fraction/cond/pred_id*
T0* 
_class
loc:@local4/local4**
_output_shapes
:	�:	�
�
,local4/zero_fraction/cond/count_nonzero/CastCast0local4/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
Truncate( *
_output_shapes
:	�*

DstT0
�
-local4/zero_fraction/cond/count_nonzero/ConstConst#^local4/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
�
5local4/zero_fraction/cond/count_nonzero/nonzero_countSum,local4/zero_fraction/cond/count_nonzero/Cast-local4/zero_fraction/cond/count_nonzero/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
local4/zero_fraction/cond/CastCast5local4/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0*
Truncate( 
�
/local4/zero_fraction/cond/count_nonzero_1/zerosConst#^local4/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
�
2local4/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual9local4/zero_fraction/cond/count_nonzero_1/NotEqual/Switch/local4/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes
:	�
�
9local4/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchlocal4/local4!local4/zero_fraction/cond/pred_id**
_output_shapes
:	�:	�*
T0* 
_class
loc:@local4/local4
�
.local4/zero_fraction/cond/count_nonzero_1/CastCast2local4/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
Truncate( *
_output_shapes
:	�*

DstT0	
�
/local4/zero_fraction/cond/count_nonzero_1/ConstConst#^local4/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
�
7local4/zero_fraction/cond/count_nonzero_1/nonzero_countSum.local4/zero_fraction/cond/count_nonzero_1/Cast/local4/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0	
�
local4/zero_fraction/cond/MergeMerge7local4/zero_fraction/cond/count_nonzero_1/nonzero_countlocal4/zero_fraction/cond/Cast*
N*
_output_shapes
: : *
T0	
�
+local4/zero_fraction/counts_to_fraction/subSublocal4/zero_fraction/Sizelocal4/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
�
,local4/zero_fraction/counts_to_fraction/CastCast+local4/zero_fraction/counts_to_fraction/sub*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
�
.local4/zero_fraction/counts_to_fraction/Cast_1Castlocal4/zero_fraction/Size*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
�
/local4/zero_fraction/counts_to_fraction/truedivRealDiv,local4/zero_fraction/counts_to_fraction/Cast.local4/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
{
local4/zero_fraction/fractionIdentity/local4/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
�
"local4/local4/local4/sparsity/tagsConst*.
value%B# Blocal4/local4/local4/sparsity*
dtype0*
_output_shapes
: 
�
local4/local4/local4/sparsityScalarSummary"local4/local4/local4/sparsity/tagslocal4/zero_fraction/fraction*
T0*
_output_shapes
: 
�
9softmax_linear/weights/Initializer/truncated_normal/shapeConst*
valueB"�   
   *)
_class
loc:@softmax_linear/weights*
dtype0*
_output_shapes
:
�
8softmax_linear/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *)
_class
loc:@softmax_linear/weights*
dtype0*
_output_shapes
: 
�
:softmax_linear/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *���;*)
_class
loc:@softmax_linear/weights
�
Csoftmax_linear/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9softmax_linear/weights/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
:	�
*

seed *
T0*)
_class
loc:@softmax_linear/weights*
seed2 
�
7softmax_linear/weights/Initializer/truncated_normal/mulMulCsoftmax_linear/weights/Initializer/truncated_normal/TruncatedNormal:softmax_linear/weights/Initializer/truncated_normal/stddev*
_output_shapes
:	�
*
T0*)
_class
loc:@softmax_linear/weights
�
3softmax_linear/weights/Initializer/truncated_normalAdd7softmax_linear/weights/Initializer/truncated_normal/mul8softmax_linear/weights/Initializer/truncated_normal/mean*
_output_shapes
:	�
*
T0*)
_class
loc:@softmax_linear/weights
�
softmax_linear/weights
VariableV2"/device:CPU:0*
shape:	�
*
dtype0*
_output_shapes
:	�
*
shared_name *)
_class
loc:@softmax_linear/weights*
	container 
�
softmax_linear/weights/AssignAssignsoftmax_linear/weights3softmax_linear/weights/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*)
_class
loc:@softmax_linear/weights*
validate_shape(*
_output_shapes
:	�

�
softmax_linear/weights/readIdentitysoftmax_linear/weights"/device:CPU:0*
T0*)
_class
loc:@softmax_linear/weights*
_output_shapes
:	�

�
'softmax_linear/biases/Initializer/ConstConst*
valueB
*    *(
_class
loc:@softmax_linear/biases*
dtype0*
_output_shapes
:

�
softmax_linear/biases
VariableV2"/device:CPU:0*
_output_shapes
:
*
shared_name *(
_class
loc:@softmax_linear/biases*
	container *
shape:
*
dtype0
�
softmax_linear/biases/AssignAssignsoftmax_linear/biases'softmax_linear/biases/Initializer/Const"/device:CPU:0*
use_locking(*
T0*(
_class
loc:@softmax_linear/biases*
validate_shape(*
_output_shapes
:

�
softmax_linear/biases/readIdentitysoftmax_linear/biases"/device:CPU:0*
T0*(
_class
loc:@softmax_linear/biases*
_output_shapes
:

�
softmax_linear/MatMulMatMullocal4/local4softmax_linear/weights/read*
T0*
_output_shapes

:
*
transpose_a( *
transpose_b( 
�
softmax_linear/softmax_linearAddsoftmax_linear/MatMulsoftmax_linear/biases/read*
_output_shapes

:
*
T0
�
<softmax_linear/softmax_linear/softmax_linear/activations/tagConst*I
value@B> B8softmax_linear/softmax_linear/softmax_linear/activations*
dtype0*
_output_shapes
: 
�
8softmax_linear/softmax_linear/softmax_linear/activationsHistogramSummary<softmax_linear/softmax_linear/softmax_linear/activations/tagsoftmax_linear/softmax_linear*
T0*
_output_shapes
: 
c
!softmax_linear/zero_fraction/SizeConst*
value	B	 R
*
dtype0	*
_output_shapes
: 
n
(softmax_linear/zero_fraction/LessEqual/yConst*
valueB	 R����*
dtype0	*
_output_shapes
: 
�
&softmax_linear/zero_fraction/LessEqual	LessEqual!softmax_linear/zero_fraction/Size(softmax_linear/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
�
(softmax_linear/zero_fraction/cond/SwitchSwitch&softmax_linear/zero_fraction/LessEqual&softmax_linear/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
�
*softmax_linear/zero_fraction/cond/switch_tIdentity*softmax_linear/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
�
*softmax_linear/zero_fraction/cond/switch_fIdentity(softmax_linear/zero_fraction/cond/Switch*
_output_shapes
: *
T0

~
)softmax_linear/zero_fraction/cond/pred_idIdentity&softmax_linear/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
�
5softmax_linear/zero_fraction/cond/count_nonzero/zerosConst+^softmax_linear/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
�
8softmax_linear/zero_fraction/cond/count_nonzero/NotEqualNotEqualAsoftmax_linear/zero_fraction/cond/count_nonzero/NotEqual/Switch:15softmax_linear/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:

�
?softmax_linear/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchsoftmax_linear/softmax_linear)softmax_linear/zero_fraction/cond/pred_id*
T0*0
_class&
$"loc:@softmax_linear/softmax_linear*(
_output_shapes
:
:

�
4softmax_linear/zero_fraction/cond/count_nonzero/CastCast8softmax_linear/zero_fraction/cond/count_nonzero/NotEqual*
Truncate( *
_output_shapes

:
*

DstT0*

SrcT0

�
5softmax_linear/zero_fraction/cond/count_nonzero/ConstConst+^softmax_linear/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
�
=softmax_linear/zero_fraction/cond/count_nonzero/nonzero_countSum4softmax_linear/zero_fraction/cond/count_nonzero/Cast5softmax_linear/zero_fraction/cond/count_nonzero/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
&softmax_linear/zero_fraction/cond/CastCast=softmax_linear/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0	
�
7softmax_linear/zero_fraction/cond/count_nonzero_1/zerosConst+^softmax_linear/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
�
:softmax_linear/zero_fraction/cond/count_nonzero_1/NotEqualNotEqualAsoftmax_linear/zero_fraction/cond/count_nonzero_1/NotEqual/Switch7softmax_linear/zero_fraction/cond/count_nonzero_1/zeros*
_output_shapes

:
*
T0
�
Asoftmax_linear/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchsoftmax_linear/softmax_linear)softmax_linear/zero_fraction/cond/pred_id*
T0*0
_class&
$"loc:@softmax_linear/softmax_linear*(
_output_shapes
:
:

�
6softmax_linear/zero_fraction/cond/count_nonzero_1/CastCast:softmax_linear/zero_fraction/cond/count_nonzero_1/NotEqual*
_output_shapes

:
*

DstT0	*

SrcT0
*
Truncate( 
�
7softmax_linear/zero_fraction/cond/count_nonzero_1/ConstConst+^softmax_linear/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
�
?softmax_linear/zero_fraction/cond/count_nonzero_1/nonzero_countSum6softmax_linear/zero_fraction/cond/count_nonzero_1/Cast7softmax_linear/zero_fraction/cond/count_nonzero_1/Const*
	keep_dims( *

Tidx0*
T0	*
_output_shapes
: 
�
'softmax_linear/zero_fraction/cond/MergeMerge?softmax_linear/zero_fraction/cond/count_nonzero_1/nonzero_count&softmax_linear/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
�
3softmax_linear/zero_fraction/counts_to_fraction/subSub!softmax_linear/zero_fraction/Size'softmax_linear/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
�
4softmax_linear/zero_fraction/counts_to_fraction/CastCast3softmax_linear/zero_fraction/counts_to_fraction/sub*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
�
6softmax_linear/zero_fraction/counts_to_fraction/Cast_1Cast!softmax_linear/zero_fraction/Size*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0	
�
7softmax_linear/zero_fraction/counts_to_fraction/truedivRealDiv4softmax_linear/zero_fraction/counts_to_fraction/Cast6softmax_linear/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
�
%softmax_linear/zero_fraction/fractionIdentity7softmax_linear/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
�
:softmax_linear/softmax_linear/softmax_linear/sparsity/tagsConst*
_output_shapes
: *F
value=B; B5softmax_linear/softmax_linear/softmax_linear/sparsity*
dtype0
�
5softmax_linear/softmax_linear/softmax_linear/sparsityScalarSummary:softmax_linear/softmax_linear/softmax_linear/sparsity/tags%softmax_linear/zero_fraction/fraction*
T0*
_output_shapes
: 
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
�
save/SaveV2/tensor_namesConst*�
value�B�
B%conv1/biases/ExponentialMovingAverageB&conv1/weights/ExponentialMovingAverageB%conv2/biases/ExponentialMovingAverageB&conv2/weights/ExponentialMovingAverageB&local3/biases/ExponentialMovingAverageB'local3/weights/ExponentialMovingAverageB&local4/biases/ExponentialMovingAverageB'local4/weights/ExponentialMovingAverageB.softmax_linear/biases/ExponentialMovingAverageB/softmax_linear/weights/ExponentialMovingAverage*
dtype0*
_output_shapes
:

w
save/SaveV2/shape_and_slicesConst*'
valueB
B B B B B B B B B B *
dtype0*
_output_shapes
:

�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesconv1/biasesconv1/weightsconv2/biasesconv2/weightslocal3/biaseslocal3/weightslocal4/biaseslocal4/weightssoftmax_linear/biasessoftmax_linear/weights*
dtypes
2

}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
_output_shapes
: *
T0
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�
B%conv1/biases/ExponentialMovingAverageB&conv1/weights/ExponentialMovingAverageB%conv2/biases/ExponentialMovingAverageB&conv2/weights/ExponentialMovingAverageB&local3/biases/ExponentialMovingAverageB'local3/weights/ExponentialMovingAverageB&local4/biases/ExponentialMovingAverageB'local4/weights/ExponentialMovingAverageB.softmax_linear/biases/ExponentialMovingAverageB/softmax_linear/weights/ExponentialMovingAverage*
dtype0*
_output_shapes
:

�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*'
valueB
B B B B B B B B B B *
dtype0*
_output_shapes
:

�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
*<
_output_shapes*
(::::::::::
�
save/AssignAssignconv1/biasessave/RestoreV2"/device:CPU:0*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(
�
save/Assign_1Assignconv1/weightssave/RestoreV2:1"/device:CPU:0*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
:@
�
save/Assign_2Assignconv2/biasessave/RestoreV2:2"/device:CPU:0*
use_locking(*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@
�
save/Assign_3Assignconv2/weightssave/RestoreV2:3"/device:CPU:0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0
�
save/Assign_4Assignlocal3/biasessave/RestoreV2:4"/device:CPU:0* 
_class
loc:@local3/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save/Assign_5Assignlocal3/weightssave/RestoreV2:5"/device:CPU:0*
T0*!
_class
loc:@local3/weights*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
save/Assign_6Assignlocal4/biasessave/RestoreV2:6"/device:CPU:0*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0* 
_class
loc:@local4/biases
�
save/Assign_7Assignlocal4/weightssave/RestoreV2:7"/device:CPU:0* 
_output_shapes
:
��*
use_locking(*
T0*!
_class
loc:@local4/weights*
validate_shape(
�
save/Assign_8Assignsoftmax_linear/biasessave/RestoreV2:8"/device:CPU:0*
use_locking(*
T0*(
_class
loc:@softmax_linear/biases*
validate_shape(*
_output_shapes
:

�
save/Assign_9Assignsoftmax_linear/weightssave/RestoreV2:9"/device:CPU:0*
validate_shape(*
_output_shapes
:	�
*
use_locking(*
T0*)
_class
loc:@softmax_linear/weights
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9"/device:CPU:0
�
Merge/MergeSummaryMergeSummaryconv1/conv1/conv1/activationsconv1/conv1/conv1/sparsityconv2/conv2/conv2/activationsconv2/conv2/conv2/sparsity local3/local3/local3/activationslocal3/local3/local3/sparsity local4/local4/local4/activationslocal4/local4/local4/sparsity8softmax_linear/softmax_linear/softmax_linear/activations5softmax_linear/softmax_linear/softmax_linear/sparsity*
N
*
_output_shapes
: 
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_73ba06812fba481ab279da525c1b2c59/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
�
save_1/SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�
Bconv1/biasesBconv1/weightsBconv2/biasesBconv2/weightsBlocal3/biasesBlocal3/weightsBlocal4/biasesBlocal4/weightsBsoftmax_linear/biasesBsoftmax_linear/weights*
dtype0*
_output_shapes
:

�
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*'
valueB
B B B B B B B B B B *
dtype0*
_output_shapes
:

�
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesconv1/biasesconv1/weightsconv2/biasesconv2/weightslocal3/biaseslocal3/weightslocal4/biaseslocal4/weightssoftmax_linear/biasessoftmax_linear/weights"/device:CPU:0*
dtypes
2

�
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: *
T0
�
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(
�
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
�
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�
Bconv1/biasesBconv1/weightsBconv2/biasesBconv2/weightsBlocal3/biasesBlocal3/weightsBlocal4/biasesBlocal4/weightsBsoftmax_linear/biasesBsoftmax_linear/weights*
dtype0*
_output_shapes
:

�
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*'
valueB
B B B B B B B B B B *
dtype0
�
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2

�
save_1/AssignAssignconv1/biasessave_1/RestoreV2"/device:CPU:0*
use_locking(*
T0*
_class
loc:@conv1/biases*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_1Assignconv1/weightssave_1/RestoreV2:1"/device:CPU:0*
use_locking(*
T0* 
_class
loc:@conv1/weights*
validate_shape(*&
_output_shapes
:@
�
save_1/Assign_2Assignconv2/biasessave_1/RestoreV2:2"/device:CPU:0*
T0*
_class
loc:@conv2/biases*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_3Assignconv2/weightssave_1/RestoreV2:3"/device:CPU:0* 
_class
loc:@conv2/weights*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0
�
save_1/Assign_4Assignlocal3/biasessave_1/RestoreV2:4"/device:CPU:0*
_output_shapes	
:�*
use_locking(*
T0* 
_class
loc:@local3/biases*
validate_shape(
�
save_1/Assign_5Assignlocal3/weightssave_1/RestoreV2:5"/device:CPU:0* 
_output_shapes
:
��*
use_locking(*
T0*!
_class
loc:@local3/weights*
validate_shape(
�
save_1/Assign_6Assignlocal4/biasessave_1/RestoreV2:6"/device:CPU:0*
use_locking(*
T0* 
_class
loc:@local4/biases*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_7Assignlocal4/weightssave_1/RestoreV2:7"/device:CPU:0*
use_locking(*
T0*!
_class
loc:@local4/weights*
validate_shape(* 
_output_shapes
:
��
�
save_1/Assign_8Assignsoftmax_linear/biasessave_1/RestoreV2:8"/device:CPU:0*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*(
_class
loc:@softmax_linear/biases
�
save_1/Assign_9Assignsoftmax_linear/weightssave_1/RestoreV2:9"/device:CPU:0*
validate_shape(*
_output_shapes
:	�
*
use_locking(*
T0*)
_class
loc:@softmax_linear/weights
�
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9"/device:CPU:0
@
save_1/restore_allNoOp^save_1/restore_shard"/device:CPU:0"&B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"8
losses.
,
local3/weight_loss:0
local4/weight_loss:0"�	
trainable_variables��
m
conv1/weights:0conv1/weights/Assignconv1/weights/read:02,conv1/weights/Initializer/truncated_normal:08
^
conv1/biases:0conv1/biases/Assignconv1/biases/read:02 conv1/biases/Initializer/Const:08
m
conv2/weights:0conv2/weights/Assignconv2/weights/read:02,conv2/weights/Initializer/truncated_normal:08
^
conv2/biases:0conv2/biases/Assignconv2/biases/read:02 conv2/biases/Initializer/Const:08
q
local3/weights:0local3/weights/Assignlocal3/weights/read:02-local3/weights/Initializer/truncated_normal:08
b
local3/biases:0local3/biases/Assignlocal3/biases/read:02!local3/biases/Initializer/Const:08
q
local4/weights:0local4/weights/Assignlocal4/weights/read:02-local4/weights/Initializer/truncated_normal:08
b
local4/biases:0local4/biases/Assignlocal4/biases/read:02!local4/biases/Initializer/Const:08
�
softmax_linear/weights:0softmax_linear/weights/Assignsoftmax_linear/weights/read:025softmax_linear/weights/Initializer/truncated_normal:08
�
softmax_linear/biases:0softmax_linear/biases/Assignsoftmax_linear/biases/read:02)softmax_linear/biases/Initializer/Const:08"�
	summaries�
�
conv1/conv1/conv1/activations:0
conv1/conv1/conv1/sparsity:0
conv2/conv2/conv2/activations:0
conv2/conv2/conv2/sparsity:0
"local3/local3/local3/activations:0
local3/local3/local3/sparsity:0
"local4/local4/local4/activations:0
local4/local4/local4/sparsity:0
:softmax_linear/softmax_linear/softmax_linear/activations:0
7softmax_linear/softmax_linear/softmax_linear/sparsity:0"�8
cond_context�8�8
�
"conv1/zero_fraction/cond/cond_text"conv1/zero_fraction/cond/pred_id:0#conv1/zero_fraction/cond/switch_t:0 *�
conv1/conv1:0
conv1/zero_fraction/cond/Cast:0
-conv1/zero_fraction/cond/count_nonzero/Cast:0
.conv1/zero_fraction/cond/count_nonzero/Const:0
8conv1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
1conv1/zero_fraction/cond/count_nonzero/NotEqual:0
6conv1/zero_fraction/cond/count_nonzero/nonzero_count:0
.conv1/zero_fraction/cond/count_nonzero/zeros:0
"conv1/zero_fraction/cond/pred_id:0
#conv1/zero_fraction/cond/switch_t:0H
"conv1/zero_fraction/cond/pred_id:0"conv1/zero_fraction/cond/pred_id:0I
conv1/conv1:08conv1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
�
$conv1/zero_fraction/cond/cond_text_1"conv1/zero_fraction/cond/pred_id:0#conv1/zero_fraction/cond/switch_f:0*�
conv1/conv1:0
/conv1/zero_fraction/cond/count_nonzero_1/Cast:0
0conv1/zero_fraction/cond/count_nonzero_1/Const:0
:conv1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
3conv1/zero_fraction/cond/count_nonzero_1/NotEqual:0
8conv1/zero_fraction/cond/count_nonzero_1/nonzero_count:0
0conv1/zero_fraction/cond/count_nonzero_1/zeros:0
"conv1/zero_fraction/cond/pred_id:0
#conv1/zero_fraction/cond/switch_f:0H
"conv1/zero_fraction/cond/pred_id:0"conv1/zero_fraction/cond/pred_id:0K
conv1/conv1:0:conv1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
�
"conv2/zero_fraction/cond/cond_text"conv2/zero_fraction/cond/pred_id:0#conv2/zero_fraction/cond/switch_t:0 *�
conv2/conv2:0
conv2/zero_fraction/cond/Cast:0
-conv2/zero_fraction/cond/count_nonzero/Cast:0
.conv2/zero_fraction/cond/count_nonzero/Const:0
8conv2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
1conv2/zero_fraction/cond/count_nonzero/NotEqual:0
6conv2/zero_fraction/cond/count_nonzero/nonzero_count:0
.conv2/zero_fraction/cond/count_nonzero/zeros:0
"conv2/zero_fraction/cond/pred_id:0
#conv2/zero_fraction/cond/switch_t:0H
"conv2/zero_fraction/cond/pred_id:0"conv2/zero_fraction/cond/pred_id:0I
conv2/conv2:08conv2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
�
$conv2/zero_fraction/cond/cond_text_1"conv2/zero_fraction/cond/pred_id:0#conv2/zero_fraction/cond/switch_f:0*�
conv2/conv2:0
/conv2/zero_fraction/cond/count_nonzero_1/Cast:0
0conv2/zero_fraction/cond/count_nonzero_1/Const:0
:conv2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
3conv2/zero_fraction/cond/count_nonzero_1/NotEqual:0
8conv2/zero_fraction/cond/count_nonzero_1/nonzero_count:0
0conv2/zero_fraction/cond/count_nonzero_1/zeros:0
"conv2/zero_fraction/cond/pred_id:0
#conv2/zero_fraction/cond/switch_f:0H
"conv2/zero_fraction/cond/pred_id:0"conv2/zero_fraction/cond/pred_id:0K
conv2/conv2:0:conv2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
�
#local3/zero_fraction/cond/cond_text#local3/zero_fraction/cond/pred_id:0$local3/zero_fraction/cond/switch_t:0 *�
local3/local3:0
 local3/zero_fraction/cond/Cast:0
.local3/zero_fraction/cond/count_nonzero/Cast:0
/local3/zero_fraction/cond/count_nonzero/Const:0
9local3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
2local3/zero_fraction/cond/count_nonzero/NotEqual:0
7local3/zero_fraction/cond/count_nonzero/nonzero_count:0
/local3/zero_fraction/cond/count_nonzero/zeros:0
#local3/zero_fraction/cond/pred_id:0
$local3/zero_fraction/cond/switch_t:0L
local3/local3:09local3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1J
#local3/zero_fraction/cond/pred_id:0#local3/zero_fraction/cond/pred_id:0
�
%local3/zero_fraction/cond/cond_text_1#local3/zero_fraction/cond/pred_id:0$local3/zero_fraction/cond/switch_f:0*�
local3/local3:0
0local3/zero_fraction/cond/count_nonzero_1/Cast:0
1local3/zero_fraction/cond/count_nonzero_1/Const:0
;local3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
4local3/zero_fraction/cond/count_nonzero_1/NotEqual:0
9local3/zero_fraction/cond/count_nonzero_1/nonzero_count:0
1local3/zero_fraction/cond/count_nonzero_1/zeros:0
#local3/zero_fraction/cond/pred_id:0
$local3/zero_fraction/cond/switch_f:0J
#local3/zero_fraction/cond/pred_id:0#local3/zero_fraction/cond/pred_id:0N
local3/local3:0;local3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
�
#local4/zero_fraction/cond/cond_text#local4/zero_fraction/cond/pred_id:0$local4/zero_fraction/cond/switch_t:0 *�
local4/local4:0
 local4/zero_fraction/cond/Cast:0
.local4/zero_fraction/cond/count_nonzero/Cast:0
/local4/zero_fraction/cond/count_nonzero/Const:0
9local4/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
2local4/zero_fraction/cond/count_nonzero/NotEqual:0
7local4/zero_fraction/cond/count_nonzero/nonzero_count:0
/local4/zero_fraction/cond/count_nonzero/zeros:0
#local4/zero_fraction/cond/pred_id:0
$local4/zero_fraction/cond/switch_t:0L
local4/local4:09local4/zero_fraction/cond/count_nonzero/NotEqual/Switch:1J
#local4/zero_fraction/cond/pred_id:0#local4/zero_fraction/cond/pred_id:0
�
%local4/zero_fraction/cond/cond_text_1#local4/zero_fraction/cond/pred_id:0$local4/zero_fraction/cond/switch_f:0*�
local4/local4:0
0local4/zero_fraction/cond/count_nonzero_1/Cast:0
1local4/zero_fraction/cond/count_nonzero_1/Const:0
;local4/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
4local4/zero_fraction/cond/count_nonzero_1/NotEqual:0
9local4/zero_fraction/cond/count_nonzero_1/nonzero_count:0
1local4/zero_fraction/cond/count_nonzero_1/zeros:0
#local4/zero_fraction/cond/pred_id:0
$local4/zero_fraction/cond/switch_f:0J
#local4/zero_fraction/cond/pred_id:0#local4/zero_fraction/cond/pred_id:0N
local4/local4:0;local4/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
�
+softmax_linear/zero_fraction/cond/cond_text+softmax_linear/zero_fraction/cond/pred_id:0,softmax_linear/zero_fraction/cond/switch_t:0 *�
softmax_linear/softmax_linear:0
(softmax_linear/zero_fraction/cond/Cast:0
6softmax_linear/zero_fraction/cond/count_nonzero/Cast:0
7softmax_linear/zero_fraction/cond/count_nonzero/Const:0
Asoftmax_linear/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
:softmax_linear/zero_fraction/cond/count_nonzero/NotEqual:0
?softmax_linear/zero_fraction/cond/count_nonzero/nonzero_count:0
7softmax_linear/zero_fraction/cond/count_nonzero/zeros:0
+softmax_linear/zero_fraction/cond/pred_id:0
,softmax_linear/zero_fraction/cond/switch_t:0d
softmax_linear/softmax_linear:0Asoftmax_linear/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Z
+softmax_linear/zero_fraction/cond/pred_id:0+softmax_linear/zero_fraction/cond/pred_id:0
�
-softmax_linear/zero_fraction/cond/cond_text_1+softmax_linear/zero_fraction/cond/pred_id:0,softmax_linear/zero_fraction/cond/switch_f:0*�
softmax_linear/softmax_linear:0
8softmax_linear/zero_fraction/cond/count_nonzero_1/Cast:0
9softmax_linear/zero_fraction/cond/count_nonzero_1/Const:0
Csoftmax_linear/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
<softmax_linear/zero_fraction/cond/count_nonzero_1/NotEqual:0
Asoftmax_linear/zero_fraction/cond/count_nonzero_1/nonzero_count:0
9softmax_linear/zero_fraction/cond/count_nonzero_1/zeros:0
+softmax_linear/zero_fraction/cond/pred_id:0
,softmax_linear/zero_fraction/cond/switch_f:0Z
+softmax_linear/zero_fraction/cond/pred_id:0+softmax_linear/zero_fraction/cond/pred_id:0f
softmax_linear/softmax_linear:0Csoftmax_linear/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0"�
	variables��
m
conv1/weights:0conv1/weights/Assignconv1/weights/read:02,conv1/weights/Initializer/truncated_normal:08
^
conv1/biases:0conv1/biases/Assignconv1/biases/read:02 conv1/biases/Initializer/Const:08
m
conv2/weights:0conv2/weights/Assignconv2/weights/read:02,conv2/weights/Initializer/truncated_normal:08
^
conv2/biases:0conv2/biases/Assignconv2/biases/read:02 conv2/biases/Initializer/Const:08
q
local3/weights:0local3/weights/Assignlocal3/weights/read:02-local3/weights/Initializer/truncated_normal:08
b
local3/biases:0local3/biases/Assignlocal3/biases/read:02!local3/biases/Initializer/Const:08
q
local4/weights:0local4/weights/Assignlocal4/weights/read:02-local4/weights/Initializer/truncated_normal:08
b
local4/biases:0local4/biases/Assignlocal4/biases/read:02!local4/biases/Initializer/Const:08
�
softmax_linear/weights:0softmax_linear/weights/Assignsoftmax_linear/weights/read:025softmax_linear/weights/Initializer/truncated_normal:08
�
softmax_linear/biases:0softmax_linear/biases/Assignsoftmax_linear/biases/read:02)softmax_linear/biases/Initializer/Const:08*�
serving_default�
3
images)
Input_Placeholder:07
scores-
softmax_linear/softmax_linear:0
tensorflow/serving/predict