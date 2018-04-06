Ngrp = 100;

XY = cell(1, Ngrp);

for gidx = 1:Ngrp
    XY{gidx} = gen_syn(randi([0,6], 1, 1), randi([40,50], 1, 1));
end

order = ENCI_pairs(XY);