function z = pagemtimes(x, y)
[x, y] = deal(halfT(x), halfT(y)); % enforce halfT type

if x.gtype || y.gtype % enforce on GPU
    [x, y] = deal(gpuArray(x), gpuArray(y));
    zc = ~(isreal(x) && isreal(y));
    zi = halfT(0);
    zr = gpu_pagemtimes_helper(real(x), real(y));
    if zc
        zr = zr - gpu_pagemtimes_helper(imag(x), imag(y)); end
    if ~isreal(y)
        zi = gpu_pagemtimes_helper(real(x), imag(y)); end
    if ~isreal(x)
        zi = zi + gpu_pagemtimes_helper(imag(x), real(y)); end
    if ~zc, z = zr; else, z = complex(zr, zi); end

else 
    % implement via mtimes
    D = max(ndims(x), ndims(y));
    xsz = size(x,1:D);
    ysz = size(y,1:D);
    zsz = [size(x,1), size(y,2), max(size(x,3:D), size(y,3:D))];
    K = prod(zsz(3:end)); % number of upper dimensions

    % find the indices
    [indz, indx, indy] = deal(cell(1,D-2));
    for k = K:-1:1
        [indz{:}] = ind2sub(zsz(3:end), k);
        [indx{:}] = arrayfun(@min,[indz{:}], xsz);
        [indy{:}] = arrayfun(@min,[indz{:}], ysz);
        ix{k} = sub2ind(xsz, indx{:});
        iy{k} = sub2ind(ysz, indy{:});
    end

    % compute
    z = halfT(zeros(zsz)); % init
    if isa(gcp('nocreate'), 'parallel.ThreadPool')
    clu = gcp(); else, clu = 0; % only use a thread pool
    end
    parfor(k = 1:K, clu)
        z(:,:,k) = mtimes(x(ix{k}), y(iy{k}));
    end
end


% TODO: convert all inputs to half
% TODO: avoid empties, but broadcast proper dimensions?

