package dut.ir.CRFJava;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * FileName： CRFParams.java <br/>
 * package：dut.ir.CRFJava <br/>
 * author：zmz and yy<br/>   
 * date：2014-7-11 <br/>
 * time：下午4:38:20 <br/>
 * email: zmz@mail.dlut.edu.cn <br/>
 * 功能：CRF的参数类，CRFTrainer类训练得到这些参数，并将其序列化到文件，CRTTagger类从文件中加载该类的一个实体，用于标记数据。 <br/>
 *     为简化代码，所有的成员变量声明为public
 */

public class CRFParams implements Serializable{
	/**
	 * 标签列表，将标签转换为其在labels中的索引，训练过程中所有的标签都用其索引来代替
	 */
	public ArrayList<String> labels = new ArrayList<String>();
	
	/**
	 * 标签的数量
	 */
	public int K = -1;
	
	/**
	 * 采用与CRF++兼容的模板格式，具体参见：<a>http://crfpp.googlecode.com/svn/trunk/doc/index.html#usage</a> <br/>
	 * 模板字典, 每个模板由模板key及其索引数组组成，如U10:%x[-2,1] 表示为("U10",[-2,1])
	 */
	public HashMap<String,Integer[]> templates;
	
	/**
	 * CRF特征函数的权值数组 <br/>
	 * 假定labels的大小为K，则每个状态特征可以生成K个特征函数，对于第i个状态特征，
	 * 其对应的K个状态特征函数的权重为：funWeight[K*i,K*i+1,..., K*i + K -1] <br/>
	 * 
	 * 同理，对于每个转移特征，则可以生成K * K个转移特征函数，对于第j个转移特征，
	 * 其对应的K * K个转移特征函数的权重为：bFunWeight[j*K*K, j*K*K, ... , j*K*K + K*K -1]<br/>
	 * 
	 * 状态特征为：Unigram，标记为'u'<br/>
	 * 转移特征为：Bigram，标记为'b'<br/>
	 */
	 public double funWeight[];
	 
	/**
	 * 每个状态特征在funWeight中的首地址映射，如：第i个特征为：'U10:%x[-2,1]'，则其对应K * i
	 */
	public HashMap<String, Integer> uFeatureAddr; 

	/**
	 * 转移特征在funWeight中的首地址映射
	 */
	public HashMap<String, Integer> bFeatureAddr;
	
	/**
	 * 句子的列数，包含标签列
	 */
	public int col = 0;
	
	public CRFParams(){
		
	}
}
