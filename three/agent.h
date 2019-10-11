#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include "weight.h"
#include <fstream>

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables
 */
class weight_agent : public agent {
public:
	weight_agent(const std::string& args = "") : agent(args) {
		if (meta.find("init") != meta.end()) // pass init=... to initialize the weight
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end()) // pass load=... to load from a specific file
			load_weights(meta["load"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end()) // pass save=... to save to a specific file
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		for(int i=0;i<32;i++)	
			net.emplace_back(16*16*16*16*16*16); // create an empty weight table with size 65536

	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
};

/**
 * base agent for agents with a learning rate
 */
class learning_agent : public agent {
public:
	learning_agent(const std::string& args = "") : agent(args), alpha(0.1f) {
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~learning_agent() {}

protected:
	float alpha;
};

class TD_agent : public weight_agent {
public:
	TD_agent(const std::string& args = "") : weight_agent("name=dummy role=player " + args), 
			alpha(0.1), leda(1), opcode({ 0, 1, 2, 3 }){
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
		if (meta.find("leda") != meta.end())
			leda = float(meta["leda"]);
	}
	uint64_t hash_value(board &b){
		uint64_t sum = 0;
		for(int i=0;i<16;i++)
			sum += ((uint64_t)b(i))<<(i*4);
		return sum;

	}
	board Uni(board target){
		board uni = target;
		uint64_t m = 0;
		for(int j=0;j<4;j++){
			uint64_t val =  hash_value(target);
			if( m < val){
				m = val;
				uni = target;
			}
			target.rotate_right();
		}
		target.reflect_horizontal();
		for(int j=0;j<4;j++){
			uint64_t val =  hash_value(target);
			if( m < val){
				m = val;
				uni = target;
			}
			target.rotate_right();
		}
		return uni;

	}
	void INC_SINGLE_Value(board& b,float inc,int index){
		int idx = 0;
		for(int i=0;i<6;i++)
			idx = (idx<<4)+b(i);
		net[index*4+0][idx] += inc;
		idx = 0;
		for(int i=0;i<6;i++)
                        idx = (idx<<4)+b(i+8);
		net[index*4+1][idx] += inc;
		idx = 0;
		for(int i=0;i<2;i++)
			for(int j=0;j<3;j++)
				idx = (idx<<4)+b[i+1][j+1];
		net[index*4+2][idx] += inc;	
		idx = 0;
		for(int i=0;i<2;i++)
			for(int j=0;j<3;j++)
				idx = (idx<<4)+b[i+2][j+1];
		net[index*4+3][idx] += inc;
		
	}

	float SINGLE_Value(board& b,int index){
		float sum = 0;
                int idx = 0;
                for(int i=0;i<6;i++)
                        idx = (idx<<4)+b(i);
                sum += net[index*4+0][idx] ;
                idx = 0;
                for(int i=0;i<6;i++)
                        idx = (idx<<4)+b(i+8);
                sum += net[index*4+1][idx];
                idx = 0;
                for(int i=0;i<2;i++)
                        for(int j=0;j<3;j++)
                                idx = (idx<<4)+b[i+1][j+1];
                sum += net[index*4+2][idx];
                idx = 0;
                for(int i=0;i<2;i++)
                        for(int j=0;j<3;j++)
                                idx = (idx<<4)+b[i+2][j+1];
                sum += net[index*4+3][idx];
		return sum;
	}
	void INC_Value(board b,float inc){
		for(int i=0;i<4;i++){
               		INC_SINGLE_Value(b,inc,i);
			b.rotate_right();
		}
		b.reflect_horizontal();
                for(int i=0;i<4;i++){
			INC_SINGLE_Value(b,inc,i+4);
			b.rotate_right();
                }

	}
	float Value(board b){
		float sum = 0;
		for(int i=0;i<4;i++){
               		sum += SINGLE_Value(b,i);
			b.rotate_right();
		}
		b.reflect_horizontal();
                for(int i=0;i<4;i++){
			sum += SINGLE_Value(b,i+4);
			b.rotate_right();
                }
		return sum;
	}
        virtual action take_action(const board& before) {
		std::vector<std::pair<action,board::reward>> legal;

                for (int op : opcode) {
                        board::reward reward = board(before).slide(op);
                        if (reward != -1) {
                                legal.push_back(std::make_pair(action::slide(op),reward));
                        }
                }
		bool first = true;
		float value = 0;	
		std::vector<std::pair<action,board::reward>> ans;

		for(auto act:legal){
			board after = before;
			act.first.apply(after);	
			float sol = act.second + Value(Uni(after));
			if( first || sol > value ){
				first = false;
				value = sol;	
				ans.clear();
				ans.push_back(act);
			} 
		}
		if( first == false ){
			auto& x = ans[rand()%ans.size()];	
			board tmp = before;
			x.first.apply(tmp);
			history.push_back(std::make_pair(tmp,x.second));	
			return x.first;
		}
                return action();
        }
	void update(){
		for(int i=history.size()-1;i>=0;i--){
			float val = (unsigned(i+1) == history.size()) ? 0 : Value(Uni(history[i+1].first));
			for(int j=i;j>=i;j--){
				val *= leda;
				if( unsigned(j+1) != history.size() )
					val += history[j].second;
				board U = Uni(history[j].first);
				float inc = (alpha * (val - Value(U)))/24;
				INC_Value(U,inc);

			}
		}
		history.clear();	
	}
	void clear(){
		history.clear();
	}


private:
	float alpha,leda;
	std::array<int, 4> opcode;
	std::vector<std::pair<board,float>> history;
		

};
	


/**
 * random environment
 * add a new random tile to an empty cell
 * 2-tile: 90%
 * 4-tile: 10%
 */
class rndenv : public random_agent {
public:
	rndenv(const std::string& args = "") : random_agent("name=random role=environment " + args),
		space({ 0, 1, 2, 3}), bags({ 1, 2, 3}), bags_idx(3), initspace({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }) {}

	virtual action take_action(const board& after) {

		if( bags_idx == 3 ) {
			std::shuffle(bags.begin(), bags.end(), engine);
			bags_idx=0;
		}
		board::cell tile = bags[bags_idx++];	
		if( after.slide_action == 4){
			std::shuffle(initspace.begin(), initspace.end(), engine);
			for (int pos : initspace) {
				if (after(pos) != 0)
					continue;
				return action::place(pos, tile);
			}
		}
		int a = after.slide_action & 1 ? 4:1;
		int b = (after.slide_action ==1 || after.slide_action == 2 ) ? 0:15;
		a = (after.slide_action ==1 || after.slide_action == 2 )? a:-a;
		if( after.slide_action >= 0){
			std::shuffle(space.begin(), space.end(), engine);
			for (int pos : space) {
				if (after(((pos*a)+b)) != 0) 
					continue;
				return action::place((pos*a)+b, tile);
			}
		}
		return action();
	}
	void clear(){
		bags_idx = 3;
	}
private:	
	std::array<int, 4> space;
	std::array<int, 3> bags;
	int bags_idx;
	std::array<int, 16> initspace;
};

/**
 * dummy player
 * select a legal action randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=dummy role=player " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) {	
				return action::slide(op);
			}
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};
