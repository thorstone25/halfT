classdef halfTTest < matlab.unittest.TestCase
    % HALFTTEST - HalfT tests class
    %
    % This class test that all halfT overloaded functions perform the
    % idential operation underneath

    properties(ClassSetupParameter)
    end

    methods(TestClassSetup, ParameterCombination = 'exhaustive')
        % Shared setup for the entire test class
        function setup(test), addpath(halfTTest.base_dir()); end
    end
    methods(TestClassTeardown)
        function teardown(test), rmpath(halfTTest.base_dir()); end
    end

    methods(TestMethodSetup)
        % Setup for each test
    end
    properties(TestParameter)
        % all functions to test
        fun = {...
            'abs','acos','acosh','alias','all','allfinite','and','any',...
            'anymissing','anynan','area','asin','asinh','atan','atan2',...
            'atanh','cast','ceil','chol','circshift','colon','complex',...
            'conj','conv','conv2','cos','cosh','cospi','ctranspose',...
            'cumsum','dealias','double','eq','exp','expm1','fft','fft2',...
            'fftn','fix','flip','floor','gather','ge','gpuArray','gt',...
            'half','halfT','ifft','ifft2','ifftn','imag','int16','int32',...
            'int64','int8','iscolumn','isempty','isequal','isequaln',...
            'isfinite','isinf','ismatrix','isnan','isreal','isrow',...
            'isscalar','issorted','isvector','ldivide','le','log','log10',...
            'log1p','log2','logical','lt','lu','max','mean','min','minus',...
            'mldivide','mod','mrdivide','mtimes','ne','not','or','permute',...
            'plus','pow10','pow2','power','prod','rdivide','real','rem',...
            'repelem','repmat','reshape','round','rsqrt','sign','sin',...
            'single','sinh','sinpi','sort','sqrt','sum','tan','tanh',...
            'times','transpose','uint16','uint32','uint64','uint8',...
            'uminus','uplus', ...
            };
        dev = {'CPU', 'mixed'}
    end
    methods(Test, ParameterCombination = 'exhaustive')
        function halfMathUniComp(test, fun, dev)
            % HALFMATHUNICOMP - Test that all unary halfT complex math 
            % functions match their half type analog

            % set of functions
            funs = {...
                'abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', ...
                'conj', 'cos', 'cosh', 'cospi', 'exp', 'expm1', 'log', ...
                'log10', 'log1p', 'log2', 'pow10', 'pow2', 'sign', ...
                'sin', 'sinh', 'sinpi', 'sqrt', 'tan', 'tanh', ...
                'uminus', 'uplus', 'imag', 'real', ...
                'ceil', 'fix', 'floor', 'round', ...
                };
            if ~ismember(fun, funs), return; end % short-circuit

            x0 = realmax('half')*(rand(10, 'like', complex(double(0)))-0.5);

            % gpu casting tests
            switch dev, case "mixed", hT = @(x)gpuArray(halfT(x)); otherwise, hT = @halfT; end
            x = hT(x0);

            % import matlab.unittest.constraints.IsEqualTo;
            % test.assertThat(x.val , IsEqualTo('Transducer'));
            f = str2func(fun);
            z0 = f(x0);
            z  = f(x );
            test.assertEqual(getfield(gather(z),'val'), z0);
            
        end
        function halfMathUniReal(test, fun, dev)
            % HALFMATHUNIREAL - Test that all unary halfT real math 
            % functions match their half type analog

            % set of functions
            funs = {...
                'rsqrt', 'complex', 'real', 'imag', 'ceil', 'fix', 'floor', 'round', 
                };
            if ~ismember(fun, funs), return; end % short-circuit

            x0 = realmax('half')*(rand(10, 'like', real(double(0)))-0.5);

            % gpu casting tests
            switch dev, case "mixed", hT = @(x)gpuArray(halfT(x)); otherwise, hT = @halfT; end
            x = hT(x0);

            % import matlab.unittest.constraints.IsEqualTo;
            % test.assertThat(x.val , IsEqualTo('Transducer'));
            f = str2func(fun);
            z0 = f(x0);
            z  = f(x );
            test.assertEqual(getfield(gather(z),'val'), z0);
        end
        function halfMathBiComp(test, fun, dev)
            % HALFMATHBICOMP - Test that all binary halfT complex math
            % functions match their half type analog

            % set of functions
            funs = {'minus','plus','power','times','conv','conv2','mod','rem','pow2','complex'};
            if ~ismember(fun, funs), return; end % short-circuit

            x0 = sqrt(realmax('half'))*(rand(10, 'like', complex(double(0)))-0.5);
            y0 = sqrt(realmax('half'))*(rand(10, 'like', complex(double(0)))-0.5);

            switch fun
                case "conv", [x0, y0] = deal(x0(:), y0(:));  % conv requires vectors
                case {'complex','mod','rem'}, [x0, y0] = deal(real(x0), real(y0)); % complex/modulus/remainder for real values only
                case {'pow2'}, y0 = half(randi([-10,10],size(y0))); % these work with integers exponent
            end

            % gpu casting types
            switch dev, case "mixed", hT = @(x)gpuArray(halfT(x)); otherwise, hT = @halfT; end
            
            x = hT(x0);
            y = hT(y0);

            % import matlab.unittest.constraints.IsEqualTo;
            % test.assertThat(x.val , IsEqualTo('Transducer'));
            f = str2func(fun);
            z0 = f(x0, y0);
            z  = f(x , y );
            test.assertEqual(getfield(gather(z),'val'), z0);
        end
        function halfMathCompComp(test, fun, dev)
            % HALFMATHCOMPCOMP - Test that all binary halfT complex math
            % comparator functions match their half type analog

            % set of functions
            funs = {'eq','ge','gt','le','lt','ne','or'};
            if ~ismember(fun, funs), return; end % short-circuit

            x0 = sqrt(realmax('half'))*(rand(10      , 'like', complex(double(0)))-0.5);
            y0 = sqrt(realmax('half'))*(rand(size(x0), 'like', complex(double(0)))-0.5);
            i = 0<randi([0,1], size(x0));
            y0(i) = x0(i);

            % gpu casting types
            switch dev, case "mixed", hT = @(x)gpuArray(halfT(x)); otherwise, hT = @halfT; end
            if fun == "or", [x0, y0] = deal(real(x0), real(y0)); end % or needs real

            x = hT(x0);
            y = hT(y0);

            % import matlab.unittest.constraints.IsEqualTo;
            % test.assertThat(x.val , IsEqualTo('Transducer'));
            f = str2func(fun);
            z0 = f(x0, y0);
            z  = f(x , y );
            test.assertEqual(gather(z), z0);
        end
        function halfNot(test, fun, dev)
            % HALFLINALGCOMP - Test that all halfT linear algebra math
            % functions match their half type analog

            % set of functions
            funs = {'not'};
            if ~ismember(fun, funs), return; end % short-circuit
            A = half(randi([0,1],10));
            switch dev, case "mixed", hT = @(x)gpuArray(halfT(x)); otherwise, hT = @halfT; end
            f = str2func(fun);
            z0 = f(A);
            z  = f(hT(A));
            test.assertEqual(gather(z), z0);
        end

        function halfLinAlgComp(test, fun, dev)
            % HALFLINALGCOMP - Test that all halfT linear algebra math 
            % functions match their half type analog

            % set of functions
            funs = {...
                'ldivide', 'rdivide', 'mtimes', 'mldivide', 'mrdivide', ...
                 'lu','chol', 'ctranspose', 'transpose' ...
                };
            if ~ismember(fun, funs), return; end % short-circuit

            A = sqrt(realmax('half')/10)*(rand(10, 'like', complex(double(0)))-0.5);
            b = sqrt(realmax('half')/10)*(rand([10, 1], 'like', complex(double(0)))-0.5);
            
            % gpu casting
            switch dev, case "mixed", hT = @(x)gpuArray(halfT(x)); otherwise, hT = @halfT; end

            % special rules
            switch fun, 
                case "chol", A = A'*A; % requires positive definite
                case 'mrdivide', b = b(1); % requires scalar divisor
                case 'mldivide', A = A(1);  % requires scalar divisor
            end

            % import matlab.unittest.constraints.IsEqualTo;
            % test.assertThat(x.val , IsEqualTo('Transducer'));
            switch fun
                case {'ldivide', 'rdivide', 'mtimes', 'mldivide', 'mrdivide'}
                    f = str2func(fun);
                    z0 = f(A, b);
                    z  = f(hT(A), hT(b));
                    test.assertEqual(getfield(gather(z),'val'), z0);
                case {'ctranspose', 'transpose', 'lu', 'chol'}
                    f = str2func(fun);
                    z0 = f(A);
                    z  = f(hT(A));
                    test.assertEqual(getfield(gather(z),'val'), z0);
            end
        end
        function halfArrAttr(test, fun, dev)
            % HALFATTR - Test that all halfT attributes match their half 
            % type analog

            % set of functions
            funs = {...
                'isempty','isscalar','iscolumn','isrow','isvector','ismatrix', ...
                'isreal','issorted', 'allfinite', 'anymissing', 'anynan', ...
                };
            if ~ismember(fun, funs), return; end % short-circuit

            switch dev, case "mixed", hT = @(x)gpuArray(halfT(x)); otherwise, hT = @halfT; end

            % set of attributes
            [sz{1}, sz{2}, sz{3}] = ndgrid([0 1 10], [0 1 10], [0 1 10]);
            sz = cell2mat(cellfun(@(s){s(:)'}, sz(:))); % 3 x S sizes
            x0 = realmax('half')*(rand(10, 'like', complex(double(0)))-0.5);
            f = str2func(fun); % function 
            switch fun
                case "isreal"
                    xc = x0; % copmlex case
                    xr = real(xc); % real case
                    test.assertEqual(gather(f(hT(xc))), f(xc));
                    test.assertEqual(gather(f(hT(xr))), f(xr));

                case "issorted"
                    xu = x0; % unsorted case
                    xs = sort(xu); % sorted case
                    test.assertEqual(gather(f(hT(xu))), f(xu));
                    test.assertEqual(gather(f(hT(xs))), f(xs));

                case {'allfinite', 'anymissing', 'anynan'}
                    xf = x0; % with neither infs nor nans
                    xi = xf; xi(0<randi([0,1],size(xf))) = inf; % with infs
                    xn = xf; xn(0<randi([0,1],size(xf))) = nan; % with nans
                    xin = xi + xf; % with both infs and nans
                    for dat = {xf, xi, xn, xin}
                        test.assertEqual(gather(f(hT(dat{1}))), f(dat{1}));
                    end


                otherwise
                    for s = 1:length(sz)
                        x0 = realmax('half')*(rand(s, 'like', complex(double(0)))-0.5);
                        x = halfT(x0);

                        % gpu casting tests
                        switch dev, case "mixed", if fun == "half", return; end, end % half type cannot be set back to GPU
                        if fun == "logical", x0 = real(x0); x = hT(x0); end % logical type cannot be real


                        % import matlab.unittest.constraints.IsEqualTo;
                        % test.assertThat(x.val , IsEqualTo('Transducer'));
                        z0 = f(x0);
                        z  = f(x );
                        switch dev,
                            case "mixed", test.assertEqual(gather(z), z0);
                            otherwise, test.assertEqual(z, z0);
                        end

                    end
            end
        end

        function halfArrShape(test, fun, dev)
            % HALFARRSHAPE - Test that all halfT reshaping operations 
            % match their half type analog

            % set of functions
            funs = {...
                'circshift', 'flip','permute', ...
                'repelem', 'repmat', 'reshape', ...
                };

            if ~ismember(fun, funs), return; end % short-circuit

            % set of attributes
            [sz{1}, sz{2}, sz{3}] = ndgrid([0 1 10], [0 1 10], [0 1 10]);
            sz = cell2mat(cellfun(@(s){s(:)'}, sz(:))); % 3 x S sizes
            switch dev, case "mixed", hT = @(x)gpuArray(halfT(x)); otherwise, hT = @halfT; end
            f = str2func(fun);

            switch fun
                case {'flip'}, opts = {2}; % choose dim 2
                case {'circshift'}, opts = {1,2}; % shift by 1 in dim 2
                case {'repmat', 'repelem'}, opts = {1,2}; % replicate with [1,2] block size
                case {'reshape'}, opts = {1,[]}; % arbitrary sizing
                case {'permute'}, opts = {[3,2,1]}; % swap dims 1 and 3
            end

            % set of attributes
            for s = 1:length(sz)
                x0 = realmax('half')*(rand(s, 'like', complex(double(0)))-0.5);
                x = hT(x0);

                % import matlab.unittest.constraints.IsEqualTo;
                % test.assertThat(x.val , IsEqualTo('Transducer'));
                z0 = f(x0, opts{:});
                z  = f(x , opts{:});
                test.assertEqual(getfield(gather(z),'val'), z0);

            end

        end
        function halfMathRedComp(test, fun, dev)
            % HALFMATHREDCOMP - Test that all halfT complex reduction math 
            % functions match their half type analog

            % set of functions
            funs = {'max', 'min', 'mean', 'sum', 'prod', 'any', 'all'};
            if ~ismember(fun, funs), return; end % short-circuit

            x0 = half(2*(rand(10, 'like', complex(double(0)))-0.5));
            
            % gpu casting tests
            switch dev, case "mixed", hT = @(x)gpuArray(halfT(x)); otherwise, hT = @halfT; end

            % operate in dimension 2
            switch fun
                case {'max', 'min'}, opts = {[], 2};
                case {'mean', 'sum', 'prod', 'any', 'all'}, opts = {2};
            end
            switch fun
                case {'any', 'all'}, x0 = x0 .* randi([0,1], [10,1]); % boolean mask
            end

            % import matlab.unittest.constraints.IsEqualTo;
            % test.assertThat(x.val , IsEqualTo('Transducer'));
            x = hT(x0);
            f = str2func(fun);
            z0 = f(x0, opts{:});
            z  = f(x , opts{:});
            if isa(z, 'halfT'),
                test.assertEqual(getfield(gather(z),'val'), z0);
            else
                test.assertEqual(gather(z), z0);
            end
            
        end
        function halfMathRedBool(test, fun, dev)
            % HALFMATHREDCOMP - Test that all halfT complex reduction math 
            % functions match their half type analog

            % set of functions
            funs = {'allfinite', 'anymissing', 'anynan'};
            if ~ismember(fun, funs), return; end % short-circuit

            x0 = half(2*(rand(10, 'like', complex(double(0)))-0.5));
            
            % gpu casting tests
            switch dev, case "mixed", hT = @(x)gpuArray(halfT(x)); otherwise, hT = @halfT; end
            x = hT(x0);

            % operate in dimension 2
            switch fun
                case {'max', 'min'}, opts = {[], 2};
                case {'mean', 'sum', 'prod'}, opts = {2};
            end

            % import matlab.unittest.constraints.IsEqualTo;
            % test.assertThat(x.val , IsEqualTo('Transducer'));
            f = str2func(fun);
            z0 = f(x0, opts{:});
            z  = f(x , opts{:});
            test.assertEqual(getfield(gather(z),'val'), z0);
            
        end
        
        function halfMathMapComp(test, fun, dev)
            % HALFMATHREDCOMP - Test that all halfT complex reduction math 
            % functions match their half type analog

            % set of functions
            funs = {'fft', 'fft2', 'fftn', 'ifft', 'ifft2', 'ifftn', 'cumsum', 'sort'};
            if ~ismember(fun, funs), return; end % short-circuit

            x0 = half(2*(rand(10, 'like', complex(double(0)))-0.5));
            
            % gpu casting tests
            switch dev, case "mixed", hT = @(x)gpuArray(halfT(x)); otherwise, hT = @halfT; end
            x = hT(x0);

            % operate in dimension 2
            switch fun
                case {'fft', 'ifft'}, opts = {[], 2};
                case {'fft2', 'ifft2', 'fftn', 'ifftn'}, opts = {};
                case {'cumsum', 'sort'}, opts = {2};
            end

            % import matlab.unittest.constraints.IsEqualTo;
            % test.assertThat(x.val , IsEqualTo('Transducer'));
            f = str2func(fun);
            z0 = f(x0, opts{:});
            z  = f(x , opts{:});
            test.assertEqual(getfield(gather(z),'val'), z0);
            
        end

        function halfTypeCast(test, fun, dev)
            % HALFMATHUNIREAL - Test that all unary halfT real math 
            % functions match their half type analog

            % set of functions
            funs = {...
                'uint16','uint32','uint64','uint8', ...
                'int16','int32','int64','int8', ...
                'double', 'single', 'half', 'logical', ...
                };
            if ~ismember(fun, funs), return; end % short-circuit

            % half type casting
            switch dev, case "mixed", hT = @(x)gpuArray(halfT(x)); otherwise, hT = @halfT; end
            x0 = realmax('half')*(rand(10, 'like', complex(double(0)))-0.5);
            x = hT(x0);

            % gpu casting tests
            switch dev, case "mixed", if fun == "half", return; end, end % half type cannot be set back to GPU
            if fun == "logical", x0 = real(x0); x = hT(x0); end % logical type cannot be real
            

            % import matlab.unittest.constraints.IsEqualTo;
            % test.assertThat(x.val , IsEqualTo('Transducer'));
            f = str2func(fun);
            z0 = f(x0);
            z  = f(x );
            switch dev,
                case "mixed", test.assertEqual(gather(z), z0);
                otherwise, test.assertEqual(z, z0);
            end
        end
                
    end
    methods(Static)
        function d = base_dir(), [d] = fullfile(fileparts(mfilename('fullpath')), '..'); end
    end
end