% HALFT - Wrapper for the half class
%
% HALFT(x) constructs a halfT object. The underlying data can be sent to a
% GPU with gpuArray, where it will be aliased as a uint16 type following
% the <a href="matlab:web('https://www.mathworks.com/matlabcentral/answers/520544-how-to-convert-half-precision-number-into-hexadecimal-representation#answer_430592')">temporary work-around on the MATLAB forum</a>.
% 
% Subsequent math calls will be forwarded to the CPU temporarily. In the
% future, native GPU calls may be added. Data movement class occur on the
% GPU.
% 
% See also HALF

classdef halfT < matlab.mixin.indexing.RedefinesParen
    properties(GetAccess=public,SetAccess=public)
        val                 % underlying data
        isaliased = false;  % whether the data is aliased
    end
    properties(Dependent, Hidden)
        gtype               % whether the data is on the GPU
    end

    methods
        % constructor
        function y = halfT(x)
            % HALFT - Construct a halfT from a native type
            %
            % y = HALFT(x) creates a halfT with the value of x. If x is a
            % gpuArray, it will be aliased as a uint16 type.
            %
            % If x is already a halfT, this has no effect.
            if isa(x, 'halfT'), 
                y = x;
            else
                y.val = half(gather(x)); % convert natively
                if isgpu(x), y = gpuArray(y); end % put back on GPU
            end
        end

    end

    % aliasing
    methods
        function y = alias(y),
            % ALIAS - Alias the data by casting it to a uint16 type.
            %
            % y = ALIAS(x) returns a halfT object y whos underlying data is of
            % type uint16. Aliased data can be sent to the GPU.
            %
            % If the data is already alaised, this function has no effect.
            %
            % See also HALFT/DEALIAS HALFT/GPUARRAY
            if ~y.isaliased,
                y.val = storedInteger(half(y.val));
                y.isaliased = true;
            end,
        end
        function y = dealias(y),
            % DEALIAS - De-alias the data by casting it back to a half type.
            %
            % y = DEALIAS(x) returns a halfT object y whos underlying data is
            % of type half. Dealiased data must reside on the CPU.
            %
            % If the data is already dealaised, this function has no effect.
            %
            % See also HALFT/ALIAS HALFT/GPUARRAY
            if y.isaliased,
                if y.gtype,
                    y.val = gather(y.val);
                end
                y.val = half.typecast(y.val); y.isaliased = false; 
            end, 
        end
        function y = gpuArray(y), y = alias(y); y.val = gpuArray(y.val); end
        % GPUARRAY - Send data to the GPU
        %
        % y = GPUARRAY(y) returns a halfT object whos underlying data
        % resides on the GPU. If the data is not already aliased, it will
        % be aliased prior to sending it to the GPU. 
        %
        % See also HALFT/ALIAS HALFT/GATHER
        function y = gather(y), y.val = gather(y.val); y = dealias(y); end
        % GPUARRAY - Return data to the CPU
        %
        % y = GPUARRAY(y) returns a halfT object whos underlying data
        % resides on the CPU. If the data is not already de-aliased, it 
        % will be de-aliased after return it to the CPU. 
        %
        % See also HALFT/ALIAS HALFT/GPUARRAY
    end

    % generic type converter
    methods
        function z = cast(y, varargin)
            % CAST - cast halfT data to a native type
            %
            % z = cast(y, NEWCLASS) returns an array z whos class matches
            % NEWCLASS
            %
            % z = cast(y, 'like', proto) returns an array z whos attributes
            % match the attributes of proto
            %
            % See also CAST
            
            if y.isaliased, z = dealias(y); else, z = y; end % dealias before casting
            z = cast(z.val, varargin{:}); % cast with MATLAB semantics
            
            % move back to GPU if only changing the class 
            % (not copying other properties)
            if numel(varargin)== 1 && y.gtype, z = gpuArray(z); end 
        end
    end

    % type conversions
    methods
        function z = double(y), z = cast(y, 'double'); end
        % cast data to a double array
        function z = single(y), z = cast(y, 'single'); end
        % cast data to a single array
        function z = half  (y), z = cast(y, 'half'  ); end
        % cast data to a half array
        function z = uint8 (y), z = cast(y, 'uint8' ); end
        % cast data to a uint8 array
        function z = uint16(y), z = cast(y, 'uint16'); end
        % cast data to a uint16 array
        function z = uint32(y), z = cast(y, 'uint32'); end
        % cast data to a uint32 array
        function z = uint64(y), z = cast(y, 'uint64'); end
        % cast data to a uint64 array
        function z =  int8 (y), z = cast(y, 'int8'  ); end
        % cast data to a int8 array
        function z =  int16(y), z = cast(y, 'int16' ); end
        % cast data to a int16 array
        function z =  int32(y), z = cast(y, 'int32' ); end
        % cast data to a int32 array
        function z =  int64(y), z = cast(y, 'int64' ); end
        % cast data to a int64 array
        function z = logical(y), z = cast(y, 'logical'); end
        % cast data to a logical array
    end

    % forward native functions
    methods(Hidden)
        % these function are identical in their GPU analog
        function x = gpuUniFun(x, fun, varargin) % unary gpu function - no casting required
            x.val = fun(x.val, varargin{:}); % apply function directly to the data
        end
        function x = gpuBiFun(x, y, fun, varargin) % binary gpu function - no casting required
            if isa(y, 'halfT'), yval = y.val; else, yval = y; end % get underlying data
            x.val = fun(x.val, yval, varargin{:}); % apply function directly to the data
        end

        % these functions run on CPU for now, but can easily be run on GPU
        % with a simple kernel
        function z = nativeUniFun(x, fun, varargin) % unary half function - casting required
            if x.gtype, x_ = gather(x); else, x_ = x; end % dealias
            z = fun(x_.val, varargin{:}); % apply function
            z = halfT(z); % return to type
            if x.gtype, z = gpuArray(z); end % move back to GPU
        end

        % these functions run on CPU for now, and require care to be run on
        % GPU with a relatively simple kernel
        function z = nativeBiFun(x, y, fun, varargin) % binary half function - casting required
            ogtype = false; % assume not on a GPU unless otherwise detected
            if isa(x,'halfT'), % x is a halfT
                ogtype = ogtype | x.gtype; % make gpuArray is x is gpuArray
                if x.isaliased, xval = getfield(dealias(x),'val'); % dealias
                else, xval = x.val; % take value directly
                end
            else, 
                ogtype = ogtype | isa(x,'gpuArray'); % make gpuArray is y is gpuArray
                xval = gather(x); % x is some native type already
            end
            if isa(y,'halfT'), % y is a halfT
                ogtype = ogtype | y.gtype; % make gpuArray is y is gpuArray
                if y.isaliased, yval = getfield(dealias(y),'val'); % dealias
                else, yval = y.val; % take value directly
                end 
            else, 
                ogtype = ogtype | isa(y,'gpuArray'); % make gpuArray is y is gpuArray
                yval = gather(y); % y is some native type already
            end
            z = fun(xval, yval, varargin{:}); % apply function
            z = halfT(z); % return to halfT type
            if ogtype, z = gpuArray(z); end % move back to GPU
        end
    end

    % overload native functions
    % TODO: use a halfT kernel to implement these in CUDA
    methods
        % arithemetic/trigonometry
        function x = abs(x), x = nativeUniFun(x, @abs); end
        function x = acos(x), x = nativeUniFun(x, @acos); end
        function x = acosh(x), x = nativeUniFun(x, @acosh); end
        function x = asin(x), x = nativeUniFun(x, @asin); end
        function x = asinh(x), x = nativeUniFun(x, @asinh); end
        function x = atan(x), x = nativeUniFun(x, @atan); end
        function x = atan2(x,y), x = nativeBiFun(x,y, @atan2); end
        function x = atanh(x), x = nativeUniFun(x, @atanh); end
        function x = conj(x), x = nativeUniFun(x, @conj); end
        function x = cos(x), x = nativeUniFun(x, @cos); end
        function x = cosh(x), x = nativeUniFun(x, @cosh); end
        function x = cospi(x), x = nativeUniFun(x, @cospi); end
        function x = exp(x), x = nativeUniFun(x, @exp); end
        function x = expm1(x), x = nativeUniFun(x, @expm1); end
        function x = log(x), x = nativeUniFun(x, @log); end
        function x = log10(x), x = nativeUniFun(x, @log10); end
        function x = log1p(x), x = nativeUniFun(x, @log1p); end
        function x = log2(x), x = nativeUniFun(x, @log2); end
        function x = minus(x,y), x = nativeBiFun(x,y,@minus); end
        function x = plus(x,y), x = nativeBiFun(x,y,@plus); end
        function x = pow10(x), x = nativeUniFun(x,@pow10); end
        function x = pow2(x,y),
            if     nargin == 1, x = nativeUniFun(x,  @pow2);
            elseif nargin == 2, x = nativeBiFun (x,y,@pow2);
            end
        end
        function x = power(x,y), x = nativeBiFun(x,y,@power); end
        function x = rsqrt(x), x = nativeUniFun(x, @rsqrt); end
        function x = sign(x), x = nativeUniFun(x, @sign); end
        function x = sin(x), x = nativeUniFun(x, @sin); end
        function x = sinh(x), x = nativeUniFun(x, @sinh); end
        function x = sinpi(x), x = nativeUniFun(x, @sinpi); end
        function x = sqrt(x), x = nativeUniFun(x, @sqrt); end
        function x = tan(x), x = nativeUniFun(x, @tan); end
        function x = tanh(x), x = nativeUniFun(x, @tanh); end
        function x = times(x,y), x = nativeBiFun(x,y,@times); end
        function x = uminus(x), x = nativeUniFun(x, @uminus); end
        function x = uplus(x), x = nativeUniFun(x, @uplus); end
        
        % complex support
        function x = complex(x,y),
            if     nargin == 1, x = gpuUniFun(x,  @complex);
            elseif nargin == 2, x = gpuBiFun (x,y,@complex);
            end
        end
        function x = imag(x), x = gpuUniFun(x, @imag); end
        function x = real(x), x = gpuUniFun(x, @real); end

        % rounding
        function x = ceil(x), x = nativeUniFun(x, @ceil); end
        function x = fix(x), x = nativeUniFun(x, @fix); end
        function x = floor(x), x = nativeUniFun(x, @floor); end
        function x = mod(x,y), x = nativeBiFun(x,y,@mod); end
        function x = rem(x,y), x = nativeBiFun(x,y,@rem); end
        function x = round(x, varargin), x = nativeUniFun(x, @round, varargin{:}); end
        function x = eps(x),   x = nativeUniFun(x, @eps); end

        % linear algebra
        function x = chol(x,varargin), x = nativeUniFun(x, @chol, varargin{:}); end
        function x = ldivide(x,y), x = nativeBiFun(x,y, @ldivide); end
        function x = lu(x, varargin), x = nativeUniFun(x, @lu, varargin{:}); end
        function x = mtimes(x,y), x = nativeBiFun(x,y,@mtimes); end
        function x = mldivide(x,y), x = nativeBiFun(x,y,@mldivide); end
        function x = mrdivide(x,y), x = nativeBiFun(x,y,@mrdivide); end
        function x = rdivide(x,y), x = nativeBiFun(x,y,@rdivide); end
        function x = ctranspose(x), x = nativeUniFun(x, @ctranspose); end
        function x = transpose(x), x = gpuUniFun(x, @transpose); end

        % logic
        function x = all(x, varargin), x = logical(nativeUniFun(x, @all, varargin{:})); end
        function x = allfinite(x), x = logical(nativeUniFun(x, @allfinite)); end
        function x = and(x,y), x = logical(nativeBiFun(x,y, @and)); end
        function x = any(x, varargin), x = logical(nativeUniFun(x, @any, varargin{:})); end
        function x = anymissing(x), x = logical(nativeUniFun(x, @anymissing)); end
        function x = anynan(x), x = logical(nativeUniFun(x, @anynan)); end
        function x = eq(x,y), x = logical(nativeBiFun(x,y, @eq)); end
        function x = ge(x,y), x = logical(nativeBiFun(x,y, @ge)); end
        function x = gt(x,y), x = logical(nativeBiFun(x,y, @gt)); end
        function x = le(x,y), x = logical(nativeBiFun(x,y, @le)); end
        function x = lt(x,y), x = logical(nativeBiFun(x,y, @lt)); end
        function x = ne(x,y), x = logical(nativeBiFun(x,y, @ne)); end
        function x = not(x),  x = logical(nativeUniFun(x, @not)); end
        function x = or(x,y), x = logical(nativeBiFun(x,y, @or)); end

        % plotting
        % function x = area(x, varargin), x = nativeUniFun(x, @area, varargin{:}); end

        % reduction
        function x = max(x, varargin), 
            if nargin >= 2 && ~isempty(varargin{1}) % second arg non-empty
                x = nativeBiFun(x, varargin{1}, @max, varargin{2:end}); % this is max(X,Y,...) syntax
            else
                x = nativeUniFun(x, @max, varargin{:}); % this is a reduction 
            end
        end
        function x = min(x, varargin),
            if nargin >= 2 && ~isempty(varargin{1}) % second arg non-empty
                x = nativeBiFun(x, varargin{1}, @min, varargin{2:end}); % this is max(X,Y,...) syntax
            else
                x = nativeUniFun(x, @min, varargin{:}); % this is a reduction
            end
        end
        function x = mean(x, varargin), x = nativeUniFun(x, @mean, varargin{:}); end
        function x = sum(x, varargin), x = nativeUniFun(x, @sum, varargin{:}); end
        function x = prod(x, varargin), x = nativeUniFun(x, @prod, varargin{:}); end
        
        % mapping
        function x = sort(x, varargin),  x = nativeUniFun(x, @sort, varargin{:}); end
        function x = cumsum(x,varargin), x = nativeUniFun(x, @cumsum,varargin{:}); end
        
        % Misc.
        function n = colon(varargin),
            valfun = @(x) {getfield(gather(x), 'val')}; % from halfT to it's value
            ishalfT = cellfun(@(x) isa(x,'halfT'), varargin); % halfT types
            isgpu = cellfun(@(x)isa(x,'gpuArray'), varargin); % gpu types
            isgpu(ishalfT) = cellfun(@(x) x.gtype, varargin); % gpu for halfT types

            % get indexing
            varargin(ishalfT) = cellfun(valfun, varargin);
            n = halfT(colon(varargin{:}));

            % return to gpu?
            if any(isgpu), n = gpuArray(n); end
        end
        
        % DSP functions
        function x = conv(x,y,varargin), x = nativeBiFun(x,y,@conv,varargin{:}); end
        function x = conv2(x,y,varargin), x = nativeBiFun(x,y,@conv2,varargin{:}); end
        function x = fft(x, varargin), x = nativeUniFun(x, @fft, varargin{:}); end
        function x = fft2(x, varargin), x = nativeUniFun(x, @fft2, varargin{:}); end
        function x = fftn(x, varargin), x = nativeUniFun(x, @fftn, varargin{:}); end
        function x = ifft(x, varargin), x = nativeUniFun(x, @ifft, varargin{:}); end
        function x = ifft2(x, varargin), x = nativeUniFun(x, @ifft2, varargin{:}); end
        function x = ifftn(x, varargin), x = nativeUniFun(x, @ifftn, varargin{:}); end
        
        % data size checks
        function x = iscolumn(x),   x = logical(gpuUniFun(x, @iscolumn)); end
        function x = isempty(x),    x = logical(gpuUniFun(x, @isempty)); end
        function x = ismatrix(x),   x = logical(gpuUniFun(x, @ismatrix)); end
        function x = isrow(x),      x = logical(gpuUniFun(x, @isrow)); end
        function x = isscalar(x),   x = logical(gpuUniFun(x, @isscalar)); end
        function x = isvector(x),   x = logical(gpuUniFun(x, @isvector)); end
        
        % data value checks
        function x = isfinite(x),   x = logical(nativeUniFun(x, @isfinite)); end
        function x = isinf(x),      x = logical(nativeUniFun(x, @isinf)); end
        function x = isnan(x),      x = logical(nativeUniFun(x, @isnan)); end
        function x = isreal(x),     x = logical(gpuUniFun(x, @isreal)); end
        function x = issorted(x,varargin), x = logical(nativeUniFun(x, @issorted, varargin{:})); end
        % function x = isnumeric(x), x = logical(nativeUniFun(x, @isnumeric)); end
        % not sure what to do with this: it may mess up certain casting
        % rules
        
        % array equivalency
        function x = isequal(x,y), x = logical(nativeBiFun(x,y, @isequal)); end
        function x = isequaln(x,y), x = logical(nativeBiFun(x,y, @isequaln)); end

        % data ordering/shaping
        function x = circshift(x,varargin), x = gpuUniFun(x, @circshift, varargin{:}); end
        function x = repelem(x, varargin),  x = gpuUniFun(x, @repelem, varargin{:}); end
        function x = repmat (x, varargin),  x = gpuUniFun(x, @repmat , varargin{:}); end
        function x = flip(x, varargin),     x = gpuUniFun(x, @flip, varargin{:}); end
        function x = permute(x, varargin),  x = gpuUniFun(x, @permute, varargin{:}); end
        function x = reshape(x, varargin),  x = gpuUniFun(x, @reshape, varargin{:}); end

        % index overloading: inferred when using redefines paren
        % function x = end(x), x = nativeUniFun(x, @end); end
        % function x = length(x), x = gpuUniFun(x, @length); end
        % function x = ndims(x), x = gpuUniFun(x, @ndims); end
        % function x = numel(x), x = gpuUniFun(x, @numel); end
        % function x = size(x, varargin), x = nativeUniFun(x, @size, varargin{:}); end
    end

    % redefines paren overloads
    methods
        function varargout = size(x, varargin), [varargout{1:nargout}] = size(x.val, varargin{:}); end
        function out = cat(dim, varargin)
            % check that all halfT values half the same aliasing.
            isaliased_ = any(cellfun(@(v)v.isaliased, varargin));
            if isaliased_, varargin = cellfun(@alias, varargin, 'UniformOutput', false); end % alias all of them to be sure
            
            % concatenate, using implicit casting - should only effect gpu
            % versus not gpu class. 
            val_ = cellfun(@(v){getfield(v,'val')}, varargin);
            out = halfT(cat(dim, val_{:}));
            out.isaliased = isaliased_;
        end
        function out = empty(varargin), out = halfT(zeros(0)); end
    end
    methods(Access=protected)
        % tell MATLAB how to index into a value
        function varargout = parenReference(x,indexOp)
            x.val = x.val.(indexOp(1)); % apply first indexing op to data
            if isscalar(indexOp),
                varargout{1} = x; % return output if last index
                return;
            else
                % Forward the other indexing operations
                [varargout{1:nargout}] = x.(indexOp(2:end));
            end
        end

        % tell MATLAB how to assign a value 
        function x = parenAssign(x,indexOp,varargin)
            if isscalar(indexOp) % final indexing operation
                assert(nargin==3); %  x(i) = y; -> (x, i, y)
                y = varargin{1}; % other halfT or numeric data

                % match aliasing if possible
                if isa(y, 'halfT') % we can sync the types
                    if x.isaliased && ~y.isaliased
                        y = alias(y); % enforce aliased halfT
                    elseif ~x.isaliased && y.isaliased
                        y = dealias(y); % enforce aliased halfT
                    end
                    x.val.(indexOp) = y.val; return; % store
                    

                elseif isnumeric(y) % we have a native numeric array
                    if x.isaliased % aliased: we'll need to apply casting here
                        yval = y;
                        if isa(y, 'gpuArray'), yval = gather(yval); end
                        yval = storedInteger(half(yval));
                        x.val.(indexOp) = yval; return; % store
                    else % not aliased: use MATLAB implicit casting rules
                        x.val.(indexOp) = y; return; % store
                    end
                else % if not, go blind - user's fault if it's messed up
                    warning("Unable to recognize input data of type " + class(y) + ".")
                    x.val.(indexOp) = y; return; 
                end
            else
                % Forward the other indexing operations
                [x.(indexOp(2:end))] = varargin{:};
            end
        end

        % tell MATLAB how many inputs to expect for this indexing operation
        function n = parenListLength(x,indexOp,ctx)
            if numel(indexOp) == 1
                n = 1;
            else
                temp = x.(indexOp(1));
                n = listLength(temp, indexOp(2:end), ctx);
            end
        end

        % tell MATLAB how to delete an indexed subset of the object
        function x = parenDelete(x,indexOp), x.val.(indexOp) = []; end

        
        %}
        
        %{
abs         barh        end         ifft2       isnumeric   lu          plot3       scatter     transpose   
acos        ceil        eq          ifftn       isreal      max         plotmatrix  scatter3    uint16      
acosh       chol        exp         imag        isrow       mean        plus        sign        uint32      
all         circshift   expm1       int16       isscalar    min         pow10       sin         uint64      
allfinite   colon       fft         int32       issorted    minus       pow2        single      uint8       
and         complex     fft2        int64       isvector    mldivide    power       sinh        uminus      
any         conj        fftn        int8        ldivide     mod         prod        sinpi       uplus       
anymissing  conv        fix         iscolumn    le          mrdivide    rdivide     size        xlim        
anynan      conv2       flip        isempty     length      mtimes      real        sort        ylim        
area        cos         floor       isequal     line        ndims       rem         sqrt        zlim        
asin        cosh        fplot       isequaln    log         ne          repelem     subsasgn    
asinh       cospi       ge          isfinite    log10       not         repmat      subsref     
atan        ctranspose  gt          isinf       log1p       numel       reshape     sum         
atan2       cumsum      half        islogical   log2        or          rgbplot     tan         
atanh       display     hypot       ismatrix    logical     permute     round       tanh        
bar         double      ifft        isnan       lt          plot        rsqrt       times       
        %}
    end

    % dependent
    methods
        function tf = get.gtype(y), tf = isgpu(y.val); end
    end
end

function tf = isgpu(x), tf = isa(x, 'gpuArray'); end