/*
 * This rundown functions You may change it to make it compatibel with other splitting funcitons.
 *
 */
#include "causalTree.h"
#include "node.h"
#include "causalTreeproto.h"

#ifdef NAN
/* NAN is supported */
#endif

void
CTA_rundown(pNode tree, int obs, double *cp, double *xpred, double *xtemp, int k, double alpha)
{
    Rprintf("CTA_rundown in CTA_rundown.c\n");
    int i, obs2 = (obs < 0) ? -(1 + obs) : obs;
    int my_leaf_id;
    pNode otree =  tree;
    pNode otree_tmp = tree;
    pNode tree_tmp = tree;
    
    int opnumber = 0;
    int j, s;
    int tmp_obs, tmp_id;
    double tr_mean, con_mean;
    double consums, trsums, cons, trs;

    /*
     * Now, repeat the following: for the cp of interest, run down the tree
     *   until I find a node with smaller complexity.  The parent node will
     *   not have collapsed, but this split will have, so this is my
     *   predictor.
     */
    Rprintf("The ct.num_unique_cp in function CTA_R in CTA_R.c is %d\n", ct.num_unique_cp);  
    for (i = 0; i < ct.num_unique_cp; i++) {
        cons = 0.;
        trs = 0.;
        consums = 0.;
        trsums = 0.;
        
        while (cp[i] < tree->complexity) {
	        tree = branch(tree, obs);
	        if (tree == 0)
		        goto oops;
	        otree = tree;
	    }
	    xpred[i] = tree->response_est[0];
        // now find other samples in the same leaf;
        my_leaf_id = tree->id;
        
        
        for (s = k; s < ct.n; s++) {
            tree_tmp = otree_tmp;
            j = ct.sorts[0][s];
            // test: 
           
            tmp_obs = (j < 0) ? -(1 + j) : j;
            while (cp[i] < tree_tmp->complexity) {
                tree_tmp = branch(tree_tmp, tmp_obs);
            }
            tmp_id = tree_tmp->id;

            if (tmp_id == my_leaf_id) {
                if (ct.treatment[j] == 0) {
                    cons += ct.wt[j];
                    consums += *ct.ydata[j] * ct.wt[j];
                } else {
                    trs += ct.wt[j];
                    trsums += *ct.ydata[j] * ct.wt[j];
                }
            }
        }
        Rprintf("before calculate in CTA_rundown.c\n");
	Rprintf("The trs in function CTA_R in CTA_R.c is %d\n", trs);  
        //calculate tr_mean and con_mean
        if (trs == 0) {
            // want to trace back to tree->parent for tr_mean;
            tr_mean = tree->parent->xtreatMean[0];
        } else {
            tr_mean = trsums / trs;
            tree->xtreatMean[0] = tr_mean;
        }
        Rprintf("The cons in function CTA_R in CTA_R.c is %d\n", cons);  
        if (cons == 0) {
            con_mean = tree->parent->xcontrolMean[0];
        } else {
            con_mean = consums / cons;
            tree->xcontrolMean[0] = con_mean;
        }
        Rprintf("pre-before CT_xeval in CTA_rundown.c\n");
        double tree_tr_mean = tree->treatMean[0];
        double tree_con_mean = tree->controlMean[0];
        Rprintf("before CT_xeval in CTA_rundown.c\n");
        xtemp[i] = (*ct_xeval)(ct.ydata[obs2], ct.wt[obs2], ct.treatment[obs2], 
                    tr_mean, con_mean, tree_tr_mean, tree_con_mean, alpha);
	Rprintf("after CT_xeval in CTA_rundown.c\n");
    }
    return;

oops:;
    if (ct.usesurrogate < 2) {  /* must have hit a missing value */
	for (i = 0; i < ct.num_unique_cp; i++)
	    xpred[i] = otree->response_est[0];

	xtemp[i] = (*ct_xeval)(ct.ydata[obs2], ct.wt[obs2], ct.treatment[obs2], tr_mean, con_mean);
	Rprintf("oops number %d.\n", opnumber++);
  return;
    }
    warning("Warning message--see rundown.c");
}
